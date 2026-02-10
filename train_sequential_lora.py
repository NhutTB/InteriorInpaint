"""
LoRA Training script for Sequential Image Editing with BrushNet + SDXL

Trains:
- UNet via LoRA adapters (~50MB output vs ~5GB full UNet)
- BrushNet fully (needs full weights since it's learning from scratch)

Benefits:
- 3-4x less VRAM vs full UNet training
- 3x faster training
- Your hybrid pipeline already supports LoRA loading via StableDiffusionXLLoraLoaderMixin

Usage:
    accelerate launch train_sequential_lora.py \
        --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
        --data_dir="path/to/dataset" \
        --output_dir="output/lora_model" \
        --train_brushnet \
        --lora_rank=16
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.utils import get_peft_model_state_dict
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("ERROR: peft not installed. Install with: pip install peft")

# Wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unets.unet_2d_condition import UNet2DConditionModel
from models.brushnet import BrushNetModel

check_min_version("0.27.0")


# Reuse dataset and helpers from main training script
from train_sequential_editing import (
    SequentialEditingDataset,
    encode_prompt,
    random_brush_mask,
    log_validation,
)


def get_lora_target_modules():
    """Get target modules for LoRA injection into SDXL UNet.
    Targets the attention layers (to_q, to_k, to_v, to_out.0)
    which is the standard approach for diffusion model LoRA."""
    return [
        "to_q", "to_k", "to_v", "to_out.0",
        # Also target cross-attention projections
        "proj_in", "proj_out",
    ]


def train_sequential_lora(args):
    """Main LoRA training function"""

    if not HAS_PEFT:
        raise ImportError("peft is required for LoRA training. Install with: pip install peft")

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== WANDB ====================

    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"lora-{os.path.basename(args.output_dir)}",
            config=vars(args),
            tags=["sequential-editing", "brushnet", "sdxl", "lora"],
        )
        print("✓ Wandb initialized")

    # ==================== LOAD MODELS ====================

    print("\n[1/7] Loading models...")

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    )

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False,
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_path or args.pretrained_model_name_or_path,
        subfolder="unet" if args.pretrained_unet_path is None else None,
    )

    if args.pretrained_brushnet_path:
        brushnet = BrushNetModel.from_pretrained(args.pretrained_brushnet_path)
    else:
        print("  → Initializing BrushNet from UNet weights...")
        brushnet = BrushNetModel.from_unet(unet)

    print("✓ Base models loaded")

    # ==================== SETUP LoRA ====================

    print("\n[2/7] Setting up LoRA...")

    # Freeze everything first
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=get_lora_target_modules(),
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)

    # Count trainable parameters
    lora_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  → LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  → UNet LoRA params: {lora_params:,} / {total_params:,} "
          f"({100 * lora_params / total_params:.2f}%)")

    # BrushNet: train fully or freeze
    if args.train_brushnet:
        brushnet.requires_grad_(True)
        brushnet_params = sum(p.numel() for p in brushnet.parameters() if p.requires_grad)
        print(f"  → BrushNet full training: {brushnet_params:,} params")
    else:
        brushnet.requires_grad_(False)
        print("  → BrushNet frozen")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_brushnet:
            brushnet.enable_gradient_checkpointing()

    print("✓ LoRA configured")

    # ==================== OPTIMIZER ====================

    print("\n[3/7] Setting up optimizer...")

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_brushnet:
        params_to_optimize.extend(list(filter(lambda p: p.requires_grad, brushnet.parameters())))

    # Optionally different LR for BrushNet vs UNet LoRA
    if args.train_brushnet and args.brushnet_learning_rate:
        optimizer = torch.optim.AdamW([
            {"params": list(filter(lambda p: p.requires_grad, unet.parameters())),
             "lr": args.learning_rate},
            {"params": list(filter(lambda p: p.requires_grad, brushnet.parameters())),
             "lr": args.brushnet_learning_rate},
        ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )

    print("✓ Optimizer ready")

    # ==================== DATASET ====================

    print("\n[4/7] Loading dataset...")

    train_dataset = SequentialEditingDataset(
        data_root=args.data_dir,
        use_sessions=args.use_sessions,
        use_global_captions=args.use_global_captions,
        use_local_captions=args.use_local_captions,
        size=args.resolution,
        center_crop=args.center_crop,
        random_mask=args.random_mask,
        augment=args.augment,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ==================== PREPARE ====================

    print("\n[5/7] Preparing training...")

    num_update_steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
    if args.max_train_steps is not None:
        max_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    if args.train_brushnet:
        unet, brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, brushnet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        brushnet.to(accelerator.device, dtype=weight_dtype)

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    print(f"✓ Training for {max_train_steps} steps ({args.num_train_epochs} epochs)")

    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.config.update({
            "dataset_size": len(train_dataset),
            "total_steps": max_train_steps,
            "lora_trainable_params": lora_params,
        })

    # ==================== RESUME ====================

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                args.resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                args.resume_from_checkpoint = None

        if args.resume_from_checkpoint:
            print(f"  Resuming from: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    # ==================== TRAINING LOOP ====================

    print("\n[6/7] Starting LoRA training...")

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="LoRA Training",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_brushnet:
            brushnet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                # Encode images
                with torch.no_grad():
                    target_latents = vae.encode(
                        batch["target"].to(dtype=vae.dtype)
                    ).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor

                    masked_latents = vae.encode(
                        batch["masked_input"].to(dtype=vae.dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor

                # Resize mask
                mask = F.interpolate(
                    batch["mask"],
                    size=(target_latents.shape[-2], target_latents.shape[-1]),
                    mode="nearest"
                )

                # Add noise
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=target_latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # Encode prompts
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    batch["prompt"],
                    [text_encoder_one, text_encoder_two],
                    [tokenizer_one, tokenizer_two],
                )
                prompt_embeds = prompt_embeds.to(target_latents.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(target_latents.device, dtype=weight_dtype)

                # SDXL add_time_ids
                add_time_ids_list = []
                for i in range(bsz):
                    orig_h = batch["original_size"][0][i].item()
                    orig_w = batch["original_size"][1][i].item()
                    add_time_ids_list.append([orig_h, orig_w, 0, 0, args.resolution, args.resolution])

                add_time_ids = torch.tensor(
                    add_time_ids_list, device=target_latents.device, dtype=weight_dtype
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # BrushNet forward
                brushnet_cond = torch.cat([masked_latents, mask.to(masked_latents.dtype)], dim=1)

                with torch.no_grad() if not args.train_brushnet else torch.enable_grad():
                    brushnet_model = accelerator.unwrap_model(brushnet) if args.train_brushnet else brushnet
                    brushnet_output = brushnet_model(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        brushnet_cond=brushnet_cond,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )

                down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet_output

                # UNet forward (with LoRA)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_add_samples=down_block_res_samples,
                    mid_block_add_sample=mid_block_res_sample,
                    up_block_add_samples=up_block_res_samples,
                    return_dict=False,
                )[0]

                # Loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Clip LoRA + BrushNet grads
                    all_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
                    if args.train_brushnet:
                        all_params.extend(list(filter(lambda p: p.requires_grad, brushnet.parameters())))
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)

                    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
                        wandb.log({
                            "train/loss": loss.detach().item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                        }, step=global_step)

                # Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # Save accelerator state
                    accelerator.save_state(save_path)

                    # Save LoRA weights separately (for easy loading in pipeline)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

                    lora_dir = os.path.join(save_path, "unet_lora")
                    os.makedirs(lora_dir, exist_ok=True)
                    torch.save(lora_state_dict, os.path.join(lora_dir, "pytorch_lora_weights.bin"))

                    # Save LoRA config
                    unwrapped_unet.peft_config["default"].save_pretrained(lora_dir)

                    if args.train_brushnet:
                        accelerator.unwrap_model(brushnet).save_pretrained(
                            os.path.join(save_path, "brushnet")
                        )

                    # Cleanup old checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                            key=lambda x: int(x.split("-")[1])
                        )
                        if len(checkpoints) > args.checkpoints_total_limit:
                            for ckpt in checkpoints[:-args.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))

                    print(f"\n✓ Checkpoint saved: {save_path}")

                # Validation
                if args.validation_steps and global_step % args.validation_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        log_validation(
                            vae, unet, brushnet,
                            [text_encoder_one, text_encoder_two],
                            [tokenizer_one, tokenizer_two],
                            train_dataset, args, accelerator, weight_dtype, global_step,
                        )

            if global_step >= max_train_steps:
                break

    # ==================== SAVE ====================

    print("\n[7/7] Saving final LoRA model...")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save LoRA weights
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

        lora_dir = os.path.join(args.output_dir, "unet_lora")
        os.makedirs(lora_dir, exist_ok=True)
        torch.save(lora_state_dict, os.path.join(lora_dir, "pytorch_lora_weights.bin"))
        unwrapped_unet.peft_config["default"].save_pretrained(lora_dir)

        if args.train_brushnet:
            accelerator.unwrap_model(brushnet).save_pretrained(
                os.path.join(args.output_dir, "brushnet")
            )

        # Save info about how to load
        info = {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "base_model": args.pretrained_model_name_or_path,
            "target_modules": get_lora_target_modules(),
            "usage": (
                "from diffusers import StableDiffusionXLPipeline\n"
                "pipe = StableDiffusionXLPipeline.from_pretrained(base_model)\n"
                "pipe.load_lora_weights('path/to/unet_lora')\n"
            ),
        }
        with open(os.path.join(args.output_dir, "lora_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"✓ LoRA weights saved to {lora_dir}")
        print(f"  Size: ~{sum(p.numel() * 2 for p in lora_state_dict.values()) / 1024 / 1024:.1f} MB")

    accelerator.end_training()

    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.finish()

    print("\n✅ LoRA Training completed!")


# ==================== ARGS ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Image Editing LoRA Training")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)
    parser.add_argument("--pretrained_brushnet_path", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank (4-64, higher=more capacity)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha, typically 2x rank")
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # BrushNet
    parser.add_argument("--train_brushnet", action="store_true",
                       help="Also train BrushNet fully alongside LoRA UNet")
    parser.add_argument("--brushnet_learning_rate", type=float, default=None,
                       help="Separate LR for BrushNet (default: same as --learning_rate)")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use_sessions", action="store_true")
    parser.add_argument("--use_global_captions", action="store_true", default=True)
    parser.add_argument("--use_local_captions", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_mask", action="store_true")
    parser.add_argument("--augment", action="store_true")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="LoRA typically uses higher LR than full fine-tuning")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts",
                                "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Validation
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--num_validation_images", type=int, default=4)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sequential-image-editing")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_images_steps", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sequential_lora(args)
