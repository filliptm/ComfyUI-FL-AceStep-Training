"""
ACE-Step Training UI Node

Main training node with rich frontend widget and WebSocket progress updates.
Uses native ComfyUI MODEL type for the ACE-Step model.
"""

import math
import os
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import Dataset, DataLoader

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    from server import PromptServer
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from ..modules.acestep_model import (
    is_acestep_model,
    get_acestep_dit,
    get_acestep_decoder,
    clone_model_for_training,
)

logger = logging.getLogger("FL_AceStep_Training")

# Discrete timesteps for turbo model (shift=3.0)
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]


def send_training_update(node_id, data):
    """Send real-time update to frontend via WebSocket."""
    if WEBSOCKET_AVAILABLE and PromptServer.instance is not None:
        PromptServer.instance.send_sync(
            "acestep.training.progress",
            {"node": str(node_id), **data}
        )


class PreprocessedTensorDataset(Dataset):
    """Dataset for loading preprocessed tensor files."""

    def __init__(self, tensor_dir: str):
        self.tensor_dir = Path(tensor_dir)
        self.samples = []

        # Load manifest
        manifest_path = self.tensor_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                self.samples = manifest.get("samples", [])
        else:
            # Scan for .pt files
            self.samples = [
                {"filename": f.name, "id": f.stem}
                for f in self.tensor_dir.glob("*.pt")
            ]

        logger.info(f"Loaded {len(self.samples)} samples from {tensor_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        # Handle both dict format (from our manifest) and string format (sdbds manifest)
        if isinstance(sample_info, dict):
            filename = sample_info["filename"]
        else:
            # sdbds stores relative paths like "./datasets/preprocessed_tensors\xyz.pt"
            # Extract just the filename to avoid path doubling
            filename = Path(sample_info).name
        tensor_path = self.tensor_dir / filename
        return torch.load(tensor_path, weights_only=False)


def collate_fn(batch):
    """Collate function for batching preprocessed tensors."""
    # For simplicity, just return the first item (batch_size=1 typically)
    if len(batch) == 1:
        item = batch[0]
        return {
            "target_latents": item["target_latents"].unsqueeze(0),
            "attention_mask": item["attention_mask"].unsqueeze(0),
            "encoder_hidden_states": item["encoder_hidden_states"].unsqueeze(0),
            "encoder_attention_mask": item["encoder_attention_mask"].unsqueeze(0),
            "context_latents": item["context_latents"].unsqueeze(0),
        }

    # Handle multiple items (variable length padding)
    max_latent_len = max(item["target_latents"].shape[0] for item in batch)
    max_enc_len = max(item["encoder_hidden_states"].shape[0] for item in batch)

    batched = {
        "target_latents": [],
        "attention_mask": [],
        "encoder_hidden_states": [],
        "encoder_attention_mask": [],
        "context_latents": [],
    }

    for item in batch:
        # Pad target latents
        t_len = item["target_latents"].shape[0]
        if t_len < max_latent_len:
            pad_len = max_latent_len - t_len
            target = F.pad(item["target_latents"], (0, 0, 0, pad_len))
            mask = F.pad(item["attention_mask"], (0, pad_len), value=0)
            context = F.pad(item["context_latents"], (0, 0, 0, pad_len))
        else:
            target = item["target_latents"]
            mask = item["attention_mask"]
            context = item["context_latents"]

        # Pad encoder sequences
        e_len = item["encoder_hidden_states"].shape[0]
        if e_len < max_enc_len:
            enc_pad = max_enc_len - e_len
            enc_hidden = F.pad(item["encoder_hidden_states"], (0, 0, 0, enc_pad))
            enc_mask = F.pad(item["encoder_attention_mask"], (0, enc_pad), value=0)
        else:
            enc_hidden = item["encoder_hidden_states"]
            enc_mask = item["encoder_attention_mask"]

        batched["target_latents"].append(target)
        batched["attention_mask"].append(mask)
        batched["encoder_hidden_states"].append(enc_hidden)
        batched["encoder_attention_mask"].append(enc_mask)
        batched["context_latents"].append(context)

    return {
        k: torch.stack(v) for k, v in batched.items()
    }


class FL_AceStep_Train:
    """
    Train LoRA

    Main training node for ACE-Step LoRA fine-tuning.

    This node:
    - Injects LoRA adapters into the DiT decoder
    - Runs the training loop with flow matching loss
    - Sends real-time progress updates to the frontend widget
    - Saves checkpoints periodically

    Training uses:
    - 8-step discrete timestep sampling (turbo model)
    - Flow matching loss: MSE(predicted_v, x1 - x0)
    - BFloat16 mixed precision
    - Gradient clipping

    Connect the Training Widget frontend to see:
    - Real-time loss graph
    - Progress bar
    - Audio preview at checkpoints
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),  # Native ComfyUI MODEL type (purple connection)
                "config": ("ACESTEP_TRAINING_CONFIG",),
                "tensor_dir": ("STRING", {
                    "default": "./output/acestep/datasets",
                    "multiline": False,
                }),
                "lora_name": ("STRING", {
                    "default": "my_lora",
                    "multiline": False,
                    "placeholder": "Name for the trained LoRA",
                }),
            },
            "optional": {
                "resume_from": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to checkpoint to resume from"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "final_lora_path", "status")
    FUNCTION = "train"
    CATEGORY = "FL AceStep/Training"
    OUTPUT_NODE = True

    # CRITICAL: This decorator MUST be here to exit ComfyUI's inference_mode context
    # before the function starts executing. Without this, all tensor operations
    # inside the function will have gradient tracking disabled.
    @torch.inference_mode(False)
    def train(self, model, config, tensor_dir, lora_name="my_lora", resume_from="", unique_id=None):
        """Run the training loop."""
        logger.info(f"Starting ACE-Step LoRA training: {lora_name}")

        # Verify this is an ACE-Step model
        if not is_acestep_model(model):
            return (model, "", "Error: Model is not an ACE-Step model")

        lora_config = config["lora"]
        training_config = config["training"]

        # Get device and dtype from the model
        device = model.load_device
        dtype = model.model.get_dtype() if hasattr(model.model, 'get_dtype') else torch.bfloat16

        # Clone the model for training to avoid modifying the original
        training_model = clone_model_for_training(model)

        # Send initial status
        send_training_update(unique_id, {
            "type": "status",
            "message": "Initializing training...",
        })

        # Create output directory using lora_name as subfolder
        # Sanitize lora_name for filesystem use
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in lora_name).strip("_") or "my_lora"
        output_dir = Path(training_config.output_dir) / safe_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        send_training_update(unique_id, {
            "type": "status",
            "message": "Loading dataset...",
        })

        dataset = PreprocessedTensorDataset(tensor_dir)
        if len(dataset) == 0:
            return (model, str(output_dir), "Error: No samples in dataset")

        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Keep in main process for ComfyUI
        )

        # Inject LoRA
        send_training_update(unique_id, {
            "type": "status",
            "message": "Injecting LoRA adapters...",
        })

        try:
            from peft import get_peft_model, LoraConfig

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
            )

            # Get decoder from the training model's diffusion_model
            dit = get_acestep_dit(training_model)
            decoder = dit.decoder

            # CRITICAL: ComfyUI loads models in inference_mode, which "taints" the parameters
            # as inference tensors. These tensors cannot participate in gradient computation
            # even after exiting inference mode.
            #
            # The param.data.clone() approach does NOT work for ComfyUI's custom modules
            # because the underlying tensors retain their inference tensor status.
            #
            # Solution: Replace ALL ComfyUI custom modules with fresh PyTorch modules.
            # This includes Linear, RMSNorm, LayerNorm, Conv1d, Embedding, etc.
            logger.info("Replacing ALL ComfyUI custom modules with fresh PyTorch modules to remove inference tensor taint...")

            # Custom RotaryEmbedding that clones buffers in forward() to avoid inference tensor taint
            # The original RotaryEmbedding returns views/slices of cached tensors which preserve inference status
            class TrainableRotaryEmbedding(torch.nn.Module):
                """RotaryEmbedding that explicitly clones cached tensors to enable gradient flow."""
                def __init__(self, original_rotary):
                    super().__init__()
                    self.dim = original_rotary.dim
                    self.base = original_rotary.base
                    self.max_position_embeddings = original_rotary.max_position_embeddings
                    self.max_seq_len_cached = original_rotary.max_seq_len_cached

                    # Clone buffers from original to fresh tensors
                    inv_freq = original_rotary.inv_freq.data.clone()
                    cos_cached = original_rotary.cos_cached.data.clone()
                    sin_cached = original_rotary.sin_cached.data.clone()

                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    self.register_buffer("cos_cached", cos_cached, persistent=False)
                    self.register_buffer("sin_cached", sin_cached, persistent=False)

                def _set_cos_sin_cache(self, seq_len, device, dtype):
                    self.max_seq_len_cached = seq_len
                    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
                    freqs = torch.outer(t, self.inv_freq)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
                    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

                def forward(self, x, seq_len=None):
                    if seq_len > self.max_seq_len_cached:
                        self._set_cos_sin_cache(seq_len, x.device, x.dtype)
                    # CRITICAL: Clone the tensors to remove inference tensor taint
                    # .to() on matching dtype/device returns a view, which preserves inference status
                    # .clone() creates a fresh tensor that can participate in autograd
                    cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).clone()
                    sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).clone()
                    return (cos, sin)

            def replace_inference_modules(module, prefix=""):
                """Replace all modules with inference-tainted weights with fresh PyTorch modules."""
                replaced = {"Linear": 0, "RMSNorm": 0, "LayerNorm": 0, "Conv1d": 0, "ConvTranspose1d": 0, "Embedding": 0, "RotaryEmbedding": 0, "Other": 0}

                for name, child in list(module.named_children()):
                    full_name = f"{prefix}.{name}" if prefix else name

                    # Check module type and create appropriate replacement
                    replaced_child = None

                    if isinstance(child, torch.nn.Linear):
                        replaced_child = torch.nn.Linear(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            device=child.weight.device,
                            dtype=child.weight.dtype
                        )
                        with torch.no_grad():
                            replaced_child.weight.copy_(child.weight.data)
                            if child.bias is not None:
                                replaced_child.bias.copy_(child.bias.data)
                        replaced["Linear"] += 1

                    elif isinstance(child, torch.nn.RMSNorm):
                        replaced_child = torch.nn.RMSNorm(
                            child.normalized_shape,
                            eps=child.eps,
                            elementwise_affine=child.weight is not None,
                            device=child.weight.device if child.weight is not None else device,
                            dtype=child.weight.dtype if child.weight is not None else dtype
                        )
                        if child.weight is not None:
                            with torch.no_grad():
                                replaced_child.weight.copy_(child.weight.data)
                        replaced["RMSNorm"] += 1

                    elif isinstance(child, torch.nn.LayerNorm):
                        replaced_child = torch.nn.LayerNorm(
                            child.normalized_shape,
                            eps=child.eps,
                            elementwise_affine=child.weight is not None,
                            device=child.weight.device if child.weight is not None else device,
                            dtype=child.weight.dtype if child.weight is not None else dtype
                        )
                        if child.weight is not None:
                            with torch.no_grad():
                                replaced_child.weight.copy_(child.weight.data)
                        if child.bias is not None:
                            with torch.no_grad():
                                replaced_child.bias.copy_(child.bias.data)
                        replaced["LayerNorm"] += 1

                    elif isinstance(child, torch.nn.Conv1d):
                        replaced_child = torch.nn.Conv1d(
                            child.in_channels,
                            child.out_channels,
                            child.kernel_size,
                            stride=child.stride,
                            padding=child.padding,
                            dilation=child.dilation,
                            groups=child.groups,
                            bias=child.bias is not None,
                            device=child.weight.device,
                            dtype=child.weight.dtype
                        )
                        with torch.no_grad():
                            replaced_child.weight.copy_(child.weight.data)
                            if child.bias is not None:
                                replaced_child.bias.copy_(child.bias.data)
                        replaced["Conv1d"] += 1

                    elif isinstance(child, torch.nn.ConvTranspose1d):
                        replaced_child = torch.nn.ConvTranspose1d(
                            child.in_channels,
                            child.out_channels,
                            child.kernel_size,
                            stride=child.stride,
                            padding=child.padding,
                            output_padding=child.output_padding,
                            groups=child.groups,
                            bias=child.bias is not None,
                            dilation=child.dilation,
                            padding_mode=child.padding_mode,
                            device=child.weight.device,
                            dtype=child.weight.dtype
                        )
                        with torch.no_grad():
                            replaced_child.weight.copy_(child.weight.data)
                            if child.bias is not None:
                                replaced_child.bias.copy_(child.bias.data)
                        replaced["ConvTranspose1d"] += 1

                    elif isinstance(child, torch.nn.Embedding):
                        replaced_child = torch.nn.Embedding(
                            child.num_embeddings,
                            child.embedding_dim,
                            padding_idx=child.padding_idx,
                            max_norm=child.max_norm,
                            norm_type=child.norm_type,
                            scale_grad_by_freq=child.scale_grad_by_freq,
                            sparse=child.sparse,
                            device=child.weight.device,
                            dtype=child.weight.dtype
                        )
                        with torch.no_grad():
                            replaced_child.weight.copy_(child.weight.data)
                        replaced["Embedding"] += 1

                    # Check for RotaryEmbedding by attribute presence (handles both ComfyUI and HF implementations)
                    elif hasattr(child, 'cos_cached') and hasattr(child, 'sin_cached') and hasattr(child, 'inv_freq'):
                        replaced_child = TrainableRotaryEmbedding(child)
                        replaced["RotaryEmbedding"] += 1
                        logger.info(f"  Replaced RotaryEmbedding at {full_name}")

                    if replaced_child is not None:
                        setattr(module, name, replaced_child)
                    else:
                        # Recursively process children
                        child_replaced = replace_inference_modules(child, full_name)
                        for k, v in child_replaced.items():
                            replaced[k] += v

                return replaced

            # Replace modules in the decoder (this is what we'll train with LoRA)
            decoder_replaced = replace_inference_modules(decoder, "decoder")
            logger.info(f"Decoder: replaced {decoder_replaced}")

            # Also replace in other DIT components that are used during forward pass
            if hasattr(dit, 'time_embed') and dit.time_embed is not None:
                te_replaced = replace_inference_modules(dit.time_embed, "time_embed")
                logger.info(f"time_embed: replaced {te_replaced}")

            if hasattr(dit, 'time_embed_r') and dit.time_embed_r is not None:
                ter_replaced = replace_inference_modules(dit.time_embed_r, "time_embed_r")
                logger.info(f"time_embed_r: replaced {ter_replaced}")

            if hasattr(dit, 'encoder') and dit.encoder is not None:
                enc_replaced = replace_inference_modules(dit.encoder, "encoder")
                logger.info(f"encoder: replaced {enc_replaced}")

            # Handle condition_embedder if it's a simple Linear
            if hasattr(dit, 'condition_embedder') and dit.condition_embedder is not None:
                if isinstance(dit.condition_embedder, torch.nn.Linear):
                    new_cond_embed = torch.nn.Linear(
                        dit.condition_embedder.in_features,
                        dit.condition_embedder.out_features,
                        bias=dit.condition_embedder.bias is not None,
                        device=dit.condition_embedder.weight.device,
                        dtype=dit.condition_embedder.weight.dtype
                    )
                    with torch.no_grad():
                        new_cond_embed.weight.copy_(dit.condition_embedder.weight.data)
                        if dit.condition_embedder.bias is not None:
                            new_cond_embed.bias.copy_(dit.condition_embedder.bias.data)
                    dit.condition_embedder = new_cond_embed
                    logger.info("condition_embedder: replaced 1 Linear module")

            logger.info("Module replacement complete (including RotaryEmbedding with clone-on-forward)")

            # Also clone ALL remaining buffers in the model as a fallback
            # Note: RotaryEmbedding buffers are now handled in the forward() via clone()
            # but we still clone other buffers just in case
            def clone_all_buffers(module, prefix=""):
                """Clone all registered buffers in a module to remove inference taint."""
                cloned_count = 0
                for name, buf in module.named_buffers(recurse=False):
                    if buf is not None:
                        # Create a fresh clone of the buffer
                        new_buf = buf.data.clone()
                        # Re-register the buffer with the cloned data
                        module.register_buffer(name, new_buf, persistent=False)
                        cloned_count += 1
                # Recursively process children
                for child_name, child in module.named_children():
                    cloned_count += clone_all_buffers(child, f"{prefix}.{child_name}" if prefix else child_name)
                return cloned_count

            # Clone buffers in the entire decoder (includes rotary embeddings)
            decoder_buffers = clone_all_buffers(decoder, "decoder")
            logger.info(f"Decoder buffers cloned: {decoder_buffers}")

            # Clone buffers in other DIT components
            if hasattr(dit, 'time_embed') and dit.time_embed is not None:
                te_buffers = clone_all_buffers(dit.time_embed, "time_embed")
                logger.info(f"time_embed buffers cloned: {te_buffers}")

            if hasattr(dit, 'time_embed_r') and dit.time_embed_r is not None:
                ter_buffers = clone_all_buffers(dit.time_embed_r, "time_embed_r")
                logger.info(f"time_embed_r buffers cloned: {ter_buffers}")

            if hasattr(dit, 'encoder') and dit.encoder is not None:
                enc_buffers = clone_all_buffers(dit.encoder, "encoder")
                logger.info(f"encoder buffers cloned: {enc_buffers}")

            # Also clone the rotary_emb buffers directly if they exist at dit level
            if hasattr(dit, 'rotary_emb') and dit.rotary_emb is not None:
                rotary_buffers = clone_all_buffers(dit.rotary_emb, "rotary_emb")
                logger.info(f"rotary_emb buffers cloned: {rotary_buffers}")

            logger.info("All module and buffer cloning complete - inference tensor taint removed")

            # Wrap decoder with LoRA
            decoder = get_peft_model(decoder, peft_config)
            decoder = decoder.to(device).to(dtype)
            decoder.train()

            # Verify LoRA parameters exist
            lora_param_count = sum(1 for n, p in decoder.named_parameters() if 'lora_' in n.lower() and p.requires_grad)
            if lora_param_count == 0:
                logger.error("No LoRA parameters found! PEFT injection may have failed.")
                return (model, str(output_dir), "Error: No LoRA parameters found after PEFT injection")

            # Update model reference
            dit.decoder = decoder

            # Freeze all non-LoRA parameters explicitly
            for name, param in decoder.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Log trainable params
            trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in decoder.parameters())
            logger.info(f"LoRA injected: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.2f}%)")

        except ImportError:
            return (model, str(output_dir), "Error: PEFT not installed")
        except Exception as e:
            logger.exception("Failed to inject LoRA")
            return (model, str(output_dir), f"Error: {str(e)}")

        # Setup optimizer - only for parameters that require gradients
        trainable_param_list = [p for p in decoder.parameters() if p.requires_grad]
        logger.info(f"Parameters passed to optimizer: {len(trainable_param_list)}")

        if len(trainable_param_list) == 0:
            logger.error("CRITICAL: No trainable parameters for optimizer!")
            return (model, str(output_dir), "Error: No trainable LoRA parameters found")

        optimizer_kwargs = {
            "lr": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
        }
        if device.type == "cuda":
            optimizer_kwargs["fused"] = True
        optimizer = AdamW(trainable_param_list, **optimizer_kwargs)

        # Setup scheduler — total_steps counts OPTIMIZER steps, not micro-batches
        steps_per_epoch = max(1, math.ceil(len(dataloader) / training_config.gradient_accumulation_steps))
        total_steps = steps_per_epoch * training_config.max_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=training_config.warmup_steps,
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - training_config.warmup_steps),
            eta_min=training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[training_config.warmup_steps],
        )

        # Set seed
        torch.manual_seed(training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(training_config.seed)

        # Timesteps tensor
        timesteps = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)

        # Training state
        global_step = 0
        start_epoch = 0
        loss_history = []
        final_checkpoint = None

        # Resume if specified
        if resume_from and os.path.exists(resume_from):
            try:
                checkpoint_info = self._load_checkpoint(resume_from, optimizer, scheduler, device)
                start_epoch = checkpoint_info.get("epoch", 0)
                global_step = checkpoint_info.get("global_step", 0)
                logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
            except Exception as e:
                logger.warning(f"Could not resume from checkpoint: {e}")

        send_training_update(unique_id, {
            "type": "status",
            "message": f"Starting training: {training_config.max_epochs} epochs",
        })

        # Progress bar
        pbar = ProgressBar(total_steps) if ProgressBar else None

        # Training loop - inference_mode(False) + enable_grad() required for ComfyUI
        with torch.inference_mode(False):
            with torch.enable_grad():
                # Initialize gradient accumulation state (matches sdbds reference)
                accumulation_step = 0
                accumulated_loss = 0.0
                optimizer.zero_grad(set_to_none=True)

                for epoch in range(start_epoch, training_config.max_epochs):
                    epoch_loss = 0.0
                    num_updates = 0

                    for batch in dataloader:
                        # Move to device
                        target_latents = batch["target_latents"].to(device).to(dtype)
                        attention_mask = batch["attention_mask"].to(device)
                        encoder_hidden_states = batch["encoder_hidden_states"].to(device).to(dtype)
                        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
                        context_latents = batch["context_latents"].to(device).to(dtype)

                        bsz = target_latents.shape[0]

                        # Sample discrete timestep
                        t_indices = torch.randint(0, len(timesteps), (bsz,), device=device)
                        t = timesteps[t_indices]  # [B]

                        # Flow matching setup
                        x0 = target_latents  # Data
                        x1 = torch.randn_like(x0)  # Noise

                        # Interpolate: x_t = t * x1 + (1 - t) * x0
                        t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                        xt = t_expanded * x1 + (1.0 - t_expanded) * x0

                        # Forward pass with mixed precision
                        try:
                            with torch.autocast(device_type='cuda', dtype=dtype):
                                decoder_outputs = dit.decoder(
                                    hidden_states=xt,
                                    timestep=t,
                                    timestep_r=t,  # Same for turbo
                                    attention_mask=attention_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    context_latents=context_latents,
                                )

                                # Compute loss INSIDE autocast (matches sdbds reference)
                                predicted_v = decoder_outputs[0] if isinstance(decoder_outputs, tuple) else decoder_outputs
                                flow = x1 - x0  # Target velocity
                                diffusion_loss = F.mse_loss(predicted_v, flow)

                            # Cast to fp32 for stable backward pass
                            loss = diffusion_loss.float()

                        except Exception as e:
                            logger.warning(f"Forward pass error: {e}")
                            continue

                        # Backward with gradient accumulation
                        scaled_loss = loss / training_config.gradient_accumulation_steps
                        scaled_loss.backward()
                        accumulated_loss += loss.item()
                        accumulation_step += 1

                        # Optimizer step when accumulation window is full
                        if accumulation_step >= training_config.gradient_accumulation_steps:
                            torch.nn.utils.clip_grad_norm_(
                                decoder.parameters(),
                                training_config.max_grad_norm
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1

                            avg_step_loss = accumulated_loss / accumulation_step
                            epoch_loss += avg_step_loss
                            num_updates += 1

                            # Track and send progress
                            loss_history.append({
                                "step": global_step,
                                "loss": avg_step_loss,
                            })
                            send_training_update(unique_id, {
                                "type": "progress",
                                "epoch": epoch + 1,
                                "total_epochs": training_config.max_epochs,
                                "step": global_step,
                                "loss": avg_step_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "loss_history": loss_history,
                            })

                            if pbar:
                                pbar.update(1)

                            accumulated_loss = 0.0
                            accumulation_step = 0

                    # Epoch end: flush remaining accumulated gradients
                    if accumulation_step > 0:
                        torch.nn.utils.clip_grad_norm_(
                            decoder.parameters(),
                            training_config.max_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        avg_step_loss = accumulated_loss / accumulation_step
                        epoch_loss += avg_step_loss
                        num_updates += 1

                        if pbar:
                            pbar.update(1)

                        accumulated_loss = 0.0
                        accumulation_step = 0

                    # Epoch complete
                    avg_loss = epoch_loss / max(num_updates, 1)
                    logger.info(f"Epoch {epoch + 1}/{training_config.max_epochs} | Loss: {avg_loss:.6f}")

                    # Save checkpoint
                    if (epoch + 1) % training_config.save_every_n_epochs == 0:
                        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
                        self._save_checkpoint(
                            decoder, optimizer, scheduler,
                            epoch + 1, global_step, checkpoint_dir
                        )
                        final_checkpoint = str(checkpoint_dir)

                        send_training_update(unique_id, {
                            "type": "checkpoint",
                            "epoch": epoch + 1,
                            "checkpoint_path": final_checkpoint,
                            "loss": avg_loss,
                        })

                # Save final checkpoint
                final_dir = output_dir / "final"
                self._save_checkpoint(
                    decoder, optimizer, scheduler,
                    training_config.max_epochs, global_step, final_dir
                )
                final_checkpoint = str(final_dir)

        # Outside grad context now

        # Unwrap PEFT from the decoder to restore clean model state
        # This prevents conflicts when ComfyUI's LoRA loader tries to patch later
        try:
            if hasattr(dit.decoder, 'merge_and_unload'):
                logger.info("Unwrapping PEFT model (merge_and_unload)...")
                dit.decoder = dit.decoder.merge_and_unload()
                dit.decoder.eval()
                logger.info("PEFT unwrapped successfully - model restored to clean state")
        except Exception as e:
            logger.warning(f"Failed to unwrap PEFT model: {e}")

        send_training_update(unique_id, {
            "type": "complete",
            "final_path": final_checkpoint,
            "total_epochs": training_config.max_epochs,
            "final_loss": avg_loss if 'avg_loss' in locals() else 0.0,
        })

        status = f"Training complete! Final checkpoint: {final_checkpoint}"
        logger.info(status)

        return (training_model, final_checkpoint, status)

    def _save_checkpoint(self, model, optimizer, scheduler, epoch, global_step, checkpoint_dir):
        """Save training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter with fixed key names for ComfyUI compatibility
        adapter_dir = checkpoint_dir / "adapter"
        adapter_dir.mkdir(exist_ok=True)

        try:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            # Get the PEFT state dict
            peft_state_dict = get_peft_model_state_dict(model)

            # Fix key names - handle double base_model.model. prefix
            # PEFT produces: base_model.model.base_model.model.layers.X...
            # ComfyUI expects: base_model.model.layers.X...
            fixed_state_dict = {}
            for key, value in peft_state_dict.items():
                new_key = key

                # Keep stripping double prefix until there's only one
                while "base_model.model.base_model.model." in new_key:
                    new_key = new_key.replace("base_model.model.base_model.model.", "base_model.model.", 1)

                fixed_state_dict[new_key] = value

            # Save in safetensors format (preferred by ComfyUI)
            save_file(fixed_state_dict, str(adapter_dir / "adapter_model.safetensors"))

            # Also save adapter_config.json for PEFT compatibility
            if hasattr(model, 'peft_config'):
                import json
                config = model.peft_config.get("default", list(model.peft_config.values())[0])
                if hasattr(config, 'to_dict'):
                    config_dict = config.to_dict()
                else:
                    config_dict = {"peft_type": "LORA", "r": 64, "lora_alpha": 128}
                # Convert sets to lists for JSON serialization
                # (PEFT returns target_modules as a set)
                for key, value in config_dict.items():
                    if isinstance(value, set):
                        config_dict[key] = sorted(list(value))
                with open(adapter_dir / "adapter_config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)

            logger.info(f"Saved LoRA adapter to {adapter_dir} ({len(fixed_state_dict)} keys)")
        except Exception as e:
            logger.warning(f"Could not save adapter: {e}")
            import traceback
            traceback.print_exc()

        # Save training state
        state_path = checkpoint_dir / "training_state.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, state_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _load_checkpoint(self, checkpoint_dir, optimizer, scheduler, device):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)

            if optimizer is not None and "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])

            return state

        return {}
