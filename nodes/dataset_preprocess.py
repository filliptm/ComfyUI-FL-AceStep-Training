"""
ACE-Step Dataset Preprocess Node (Staged + Tiled + Dimension Fix)

Fixes:
- Added .squeeze(0) to target_latents to prevent "got 3 and 4" dimension error during training.
- Keeps all previous Low VRAM optimizations (Staged + Tiled).
"""

import json
import logging
import random
from pathlib import Path
import gc
import math

import torch

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_silence_latent,
    get_acestep_encoder,
)
from ..modules.audio_utils import load_audio, vae_encode_direct

logger = logging.getLogger("FL_AceStep_Training")

# SFT generation prompt template
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

_REFER_AUDIO_CACHE: dict = {}

def force_clear_memory():
    """Aggressively clear VRAM and RAM"""
    if model_management:
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def _get_refer_audio_tensors(device, dtype):
    cache_key = (device, dtype)
    if cache_key not in _REFER_AUDIO_CACHE:
        _REFER_AUDIO_CACHE[cache_key] = (
            torch.zeros(1, 1, 64, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=torch.long),
        )
    refer_audio_hidden, refer_audio_order_mask = _REFER_AUDIO_CACHE[cache_key]
    refer_audio_hidden.zero_()
    refer_audio_order_mask.zero_()
    return refer_audio_hidden, refer_audio_order_mask

def encode_text_and_lyrics(clip, text: str, lyrics: str, device, dtype):
    tokens = clip.tokenize(text, lyrics=lyrics)
    result = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

    text_hidden_states = result["cond"].to(device=device, dtype=dtype, non_blocking=True)
    text_attention_mask = torch.ones(
        text_hidden_states.shape[:2], device=device, dtype=dtype
    )

    lyric_hidden_states = result.get("conditioning_lyrics", None)
    if lyric_hidden_states is not None:
        lyric_hidden_states = lyric_hidden_states.to(device=device, dtype=dtype, non_blocking=True)
        if lyric_hidden_states.dim() == 2:
            lyric_hidden_states = lyric_hidden_states.unsqueeze(0)
        lyric_attention_mask = torch.ones(
            lyric_hidden_states.shape[:2], device=device, dtype=dtype
        )
    else:
        lyric_hidden_states = torch.zeros(1, 1, text_hidden_states.shape[-1],
                                          device=device, dtype=dtype)
        lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask


def vae_encode_tiled(vae_model, waveform, device, dtype, chunk_seconds=20, sr=44100):
    """
    Encodes audio in chunks to save VRAM.
    """
    chunk_size = int(chunk_seconds * sr)
    total_samples = waveform.shape[-1]
    
    latents_list = []
    
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        
        audio_chunk = waveform[:, start:end].unsqueeze(0).to(device=device, dtype=dtype)
        
        with torch.no_grad():
            chunk_latents = vae_encode_direct(vae_model, audio_chunk, device, dtype)
        
        latents_list.append(chunk_latents.cpu())
        
        del audio_chunk, chunk_latents
    
    if not latents_list:
        return None
        
    full_latents = torch.cat(latents_list, dim=1)
    return full_latents


class FL_AceStep_PreprocessDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "output_dir": ("STRING", {
                    "default": "./output/acestep/datasets",
                    "multiline": False,
                }),
                "low_vram": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "Enabled (Staged + Tiled)", 
                    "label_off": "Disabled (Standard)"
                }),
            },
            "optional": {
                "max_duration": ("FLOAT", {
                    "default": 240.0, "min": 10.0, "max": 600.0, "step": 10.0,
                }),
                "genre_ratio": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 5,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_path", "sample_count", "status")
    FUNCTION = "preprocess"
    CATEGORY = "FL AceStep/Dataset"
    OUTPUT_NODE = True

    def preprocess(self, dataset, model, vae, clip, output_dir, low_vram=True, max_duration=240.0, genre_ratio=0):
        # Initial Cleanup
        force_clear_memory()
        
        samples = dataset.samples
        if not samples: return (output_dir, 0, "No samples")

        if not is_acestep_model(model):
            return (output_dir, 0, "Error: Model is not an ACE-Step model")

        labeled_samples = [s for s in samples if s.labeled or s.caption]
        if not labeled_samples: return (output_dir, 0, "No labeled samples")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        device = model_management.get_torch_device() if model_management else torch.device('cuda')
        
        # Temp Storage
        temp_storage = {s.id: {'sample': s} for s in labeled_samples}
        
        vae_model = vae.first_stage_model
        vae_dtype = vae.vae_dtype
        
        condition_encoder = get_acestep_encoder(model)
        enc_param = next(condition_encoder.parameters())
        enc_dtype = enc_param.dtype
        
        silence_latent = get_silence_latent(model)
        if silence_latent is None:
            silence_latent = torch.zeros(1, 750, 64, device=device, dtype=enc_dtype)

        processed_count = 0
        manifest = []
        errors = []
        
        total_steps = len(labeled_samples) * 3
        pbar = ProgressBar(total_steps) if ProgressBar else None

        logger.info(f"Starting Preprocessing (Low VRAM: {low_vram}) for {len(labeled_samples)} samples.")

        try:
            # =================================================================================
            # STAGE 1: VAE PROCESSING
            # =================================================================================
            logger.info("--- STAGE 1/3: VAE Encoding (Tiled) ---")
            
            if low_vram: 
                model_management.load_models_gpu([vae.patcher])
            
            with torch.inference_mode():
                for i, sample in enumerate(labeled_samples):
                    try:
                        waveform, sr = load_audio(sample.audio_path, max_duration=max_duration)
                        
                        if low_vram:
                            target_latents = vae_encode_tiled(
                                vae_model, waveform, device, vae_dtype, chunk_seconds=20, sr=sr
                            )
                        else:
                            audio = waveform.unsqueeze(0).to(device=device, dtype=vae_dtype, non_blocking=True)
                            target_latents = vae_encode_direct(vae_model, audio, device, vae_dtype).cpu()
                            del audio

                        if target_latents is not None:
                            temp_storage[sample.id]['target_latents'] = target_latents
                        
                        del waveform
                    except Exception as e:
                        logger.error(f"VAE Error on {sample.id}: {e}")
                        errors.append(f"{sample.id} (VAE): {e}")
                    
                    if pbar: pbar.update(1)
            
            if low_vram: force_clear_memory()

            # =================================================================================
            # STAGE 2: CLIP PROCESSING
            # =================================================================================
            logger.info("--- STAGE 2/3: CLIP Encoding ---")
            if low_vram: model_management.load_models_gpu([clip.patcher])

            with torch.inference_mode():
                for i, sample in enumerate(labeled_samples):
                    if sample.id not in temp_storage or 'target_latents' not in temp_storage[sample.id]:
                        if pbar: pbar.update(1)
                        continue

                    try:
                        caption = sample.caption
                        custom_tag = dataset.metadata.custom_tag
                        tag_position = dataset.metadata.tag_position
                        
                        if custom_tag:
                            if tag_position == "prepend": caption = f"{custom_tag}, {caption}"
                            elif tag_position == "append": caption = f"{caption}, {custom_tag}"
                            elif tag_position == "replace": caption = custom_tag

                        use_genre = random.randint(0, 100) < genre_ratio and sample.genre
                        text_content = sample.genre if use_genre else caption
                        
                        temp_storage[sample.id]['caption'] = caption
                        temp_storage[sample.id]['final_lyrics'] = sample.lyrics if sample.lyrics else "[Instrumental]"

                        metas_str = (
                            f"- bpm: {sample.bpm if sample.bpm else 'N/A'}\n"
                            f"- timesignature: {sample.timesignature if sample.timesignature else 'N/A'}\n"
                            f"- keyscale: {sample.keyscale if sample.keyscale else 'N/A'}\n"
                            f"- duration: {int(sample.duration)} seconds\n"
                        )
                        text_prompt = SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, text_content, metas_str)

                        ths, tam, lhs, lam = encode_text_and_lyrics(
                            clip, text_prompt, temp_storage[sample.id]['final_lyrics'], device, enc_dtype
                        )
                        
                        temp_storage[sample.id]['text_hidden_states'] = ths.cpu()
                        temp_storage[sample.id]['text_attention_mask'] = tam.cpu()
                        temp_storage[sample.id]['lyric_hidden_states'] = lhs.cpu()
                        temp_storage[sample.id]['lyric_attention_mask'] = lam.cpu()
                        
                    except Exception as e:
                        logger.error(f"CLIP Error on {sample.id}: {e}")
                        errors.append(f"{sample.id} (CLIP): {e}")

                    if pbar: pbar.update(1)

            if low_vram: force_clear_memory()

            # =================================================================================
            # STAGE 3: CONDITION ENCODER & SAVE
            # =================================================================================
            logger.info("--- STAGE 3/3: Condition Encoding & Saving ---")
            
            if low_vram:
                condition_encoder.to(device)
            
            with torch.inference_mode():
                for i, sample in enumerate(labeled_samples):
                    sid = sample.id
                    if sid not in temp_storage or 'text_hidden_states' not in temp_storage[sid]:
                        if pbar: pbar.update(1)
                        continue

                    try:
                        data = temp_storage[sid]
                        
                        ths = data['text_hidden_states'].to(device)
                        tam = data['text_attention_mask'].to(device)
                        lhs = data['lyric_hidden_states'].to(device)
                        lam = data['lyric_attention_mask'].to(device)
                        
                        refer_audio_hidden, refer_audio_order_mask = _get_refer_audio_tensors(device, enc_dtype)

                        encoder_hidden_states, encoder_attention_mask = condition_encoder(
                            text_hidden_states=ths,
                            text_attention_mask=tam,
                            lyric_hidden_states=lhs,
                            lyric_attention_mask=lam,
                            refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                            refer_audio_order_mask=refer_audio_order_mask,
                        )

                        target_latents = data['target_latents']
                        latent_length = target_latents.shape[1]
                        
                        context_latents = torch.empty((1, latent_length, 128), device=device, dtype=enc_dtype)
                        
                        src = silence_latent.to(dtype=enc_dtype)
                        src_len = src.shape[1]
                        take = min(latent_length, src_len)
                        context_latents[:, :take, :64] = src[:, :take, :]
                        if take < latent_length:
                            remaining = latent_length - take
                            pos = take
                            while remaining > 0:
                                chunk = min(remaining, src_len)
                                context_latents[:, pos:pos + chunk, :64] = src[:, :chunk, :]
                                pos += chunk
                                remaining -= chunk
                        context_latents[:, :, 64:] = 1

                        tensor_data = {
                            # --- CRITICAL FIX: SQUEEZE BATCH DIMENSION ---
                            "target_latents": target_latents.squeeze(0).cpu(), 
                            "attention_mask": torch.ones(latent_length).cpu(), # Squeeze implicitly done by not creating batch dim
                            "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
                            "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
                            "context_latents": context_latents.squeeze(0).cpu(),
                            # ---------------------------------------------
                            "metadata": {
                                "audio_path": sample.audio_path,
                                "filename": sample.filename,
                                "caption": data['caption'],
                                "lyrics": data['final_lyrics'],
                                "duration": sample.duration,
                                "bpm": sample.bpm,
                                "keyscale": sample.keyscale,
                                "timesignature": sample.timesignature,
                                "language": sample.language,
                                "is_instrumental": sample.is_instrumental,
                            }
                        }

                        tensor_filename = f"{sample.id}.pt"
                        tensor_path = output_path / tensor_filename
                        torch.save(tensor_data, tensor_path)

                        manifest.append({
                            "id": sample.id,
                            "filename": tensor_filename,
                            "audio_path": sample.audio_path,
                            "caption": sample.caption,
                            "duration": sample.duration,
                            "bpm": sample.bpm,
                            "keyscale": sample.keyscale,
                            "is_instrumental": sample.is_instrumental,
                        })

                        processed_count += 1
                        logger.info(f"Saved {sample.filename}")
                        
                        del ths, tam, lhs, lam, encoder_hidden_states, encoder_attention_mask, context_latents

                    except Exception as e:
                        logger.error(f"Save Error on {sid}: {e}")
                        errors.append(f"{sid} (Save): {e}")

                    if pbar: pbar.update(1)

            # Final Cleanup
            condition_encoder.to("cpu")
            force_clear_memory()
            del temp_storage

        except Exception as e:
            logger.critical(f"Critical Pipeline Error: {e}")
            return (output_dir, 0, f"Critical Error: {e}")

        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "samples": manifest,
                "metadata": {
                    "total_samples": processed_count,
                    "max_duration": max_duration,
                    "genre_ratio": genre_ratio,
                    "custom_tag": dataset.metadata.custom_tag,
                }
            }, f, indent=2, ensure_ascii=False)

        status = f"Preprocessed {processed_count}/{len(labeled_samples)} samples"
        if errors:
            status += f" ({len(errors)} errors)"

        logger.info(status)
        return (str(output_path), processed_count, status)
