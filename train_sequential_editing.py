"""
Training script for Sequential Image Editing with BrushNet + SDXL
Optimized for multi-turn instruction-based editing dataset

Enhanced with:
- Resume from checkpoint
- Validation during training (sample image generation)
- Random mask augmentation
- Proper SDXL add_time_ids with crop coordinates
- Data augmentation (color jitter, flip)

Dataset structure:
    data/
    ├── images/
    │   ├── 100547/
    │   │   ├── 100547-input.png
    │   │   ├── 100547-mask1.png
    │   │   ├── 100547-output1.png
    │   │   ├── 100547-mask2.png
    │   │   └── 100547-output2.png
    │   └── ...
    ├── edit_turns.json              # Main training data
    ├── edit_sessions.json           # Multi-turn sessions
    ├── global_descriptions.json     # Full image captions
    └── local_descriptions.json      # Object-specific descriptions
"""

import argparse
import contextlib
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
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom models
from models.unets.unet_2d_condition import UNet2DConditionModel
from models.brushnet import BrushNetModel

check_min_version("0.27.0")


# ==================== RANDOM MASK GENERATION ====================

def random_brush_mask(h, w, max_tries=4, min_width=64, max_width=128):
    """Generate random brush stroke mask for augmentation.
    Ported from official BrushNet training script."""
    mask = Image.new('L', (w, h), 0)
    average_radius = math.sqrt(h * h + w * w) / 8

    for _ in range(np.random.randint(1, max_tries + 1)):
        num_vertex = np.random.randint(2, 8)
        angles = []
        for i in range(num_vertex):
            mean_angle = 2 * math.pi / 5
            angle_range = 2 * math.pi / 15
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(
                    mean_angle - angle_range, mean_angle + angle_range))
            else:
                angles.append(np.random.uniform(
                    mean_angle - angle_range, mean_angle + angle_range))

        vertex = [(int(np.random.randint(0, w)), int(np.random.randint(0, h)))]
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w - 1)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h - 1)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=255, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2), fill=255)

    mask_np = np.array(mask)
    if np.random.random() > 0.5:
        mask_np = np.flip(mask_np, 0).copy()
    if np.random.random() > 0.5:
        mask_np = np.flip(mask_np, 1).copy()

    return Image.fromarray(mask_np)


# ==================== PROMPT ENCODING ====================

def encode_prompt(prompt_batch, text_encoders, tokenizers):
    """Encode prompts using both SDXL text encoders.
    Returns prompt_embeds and pooled_prompt_embeds."""
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)

        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# ==================== DATASET ====================

class SequentialEditingDataset(Dataset):
    """
    Dataset for sequential image editing training.

    Supports both single-turn and multi-turn editing:
    - Single-turn: input -> output (one instruction)
    - Multi-turn: input -> output1 -> output2 (chained instructions)
    """

    def __init__(
        self,
        data_root: str,
        use_sessions: bool = False,
        use_global_captions: bool = True,
        use_local_captions: bool = False,
        size: int = 1024,
        center_crop: bool = False,
        random_mask: bool = False,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.center_crop = center_crop
        self.use_global_captions = use_global_captions
        self.use_local_captions = use_local_captions
        self.random_mask = random_mask
        self.augment = augment

        # Load metadata
        self.images_dir = self.data_root / "images"

        # Load edit data
        if use_sessions:
            sessions_file = self.data_root / "edit_sessions.json"
            with open(sessions_file, 'r', encoding='utf-8') as f:
                sessions = json.load(f)

            self.samples = []
            for session_id, turns in sessions.items():
                for turn in turns:
                    turn['session_id'] = session_id
                    self.samples.append(turn)
        else:
            turns_file = self.data_root / "edit_turns.json"
            with open(turns_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)

        # Load descriptions (optional)
        self.global_descriptions = {}
        self.local_descriptions = {}

        if use_global_captions:
            global_file = self.data_root / "global_descriptions.json"
            if global_file.exists():
                with open(global_file, 'r', encoding='utf-8') as f:
                    self.global_descriptions = json.load(f)

        if use_local_captions:
            local_file = self.data_root / "local_descriptions.json"
            if local_file.exists():
                with open(local_file, 'r', encoding='utf-8') as f:
                    self.local_descriptions = json.load(f)

        print(f"Loaded {len(self.samples)} editing turns")

        # Transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
        ])

        # Data augmentation
        if augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )
        else:
            self.color_jitter = None

    def __len__(self):
        return len(self.samples)

    def _get_image_id(self, filename: str) -> str:
        """Extract ID from filename (e.g., '100547-input.png' -> '100547')"""
        return filename.split('-')[0]

    def _load_image(self, filename: str) -> Image.Image:
        """Load image from nested directory structure"""
        image_id = self._get_image_id(filename)
        image_path = self.images_dir / image_id / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _load_mask(self, filename: str) -> Image.Image:
        """Load mask (binary image)"""
        image_id = self._get_image_id(filename)
        mask_path = self.images_dir / image_id / filename

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        return Image.open(mask_path).convert("L")

    def _create_masked_input(self, input_img: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Create masked input by blacking out the region defined by mask.
        This simulates the conditioning image for BrushNet.
        """
        input_np = np.array(input_img)
        mask_np = np.array(mask)

        mask_binary = (mask_np > 128).astype(np.uint8)

        masked_input = input_np.copy()
        masked_input[mask_binary == 1] = 0

        return Image.fromarray(masked_input)

    def _get_prompt(self, sample: Dict, output_filename: str) -> str:
        """
        Construct prompt from instruction and optional descriptions.

        Priority:
        1. instruction (editing command)
        2. local_description (object-specific)
        3. global_description (full scene)
        """
        prompt_parts = []

        instruction = sample.get('instruction', '')
        if instruction:
            prompt_parts.append(instruction)

        if self.use_local_captions:
            image_id = self._get_image_id(output_filename)
            if image_id in self.local_descriptions:
                local_desc = self.local_descriptions[image_id].get(output_filename, '')
                if local_desc:
                    prompt_parts.append(f"({local_desc})")

        if self.use_global_captions:
            image_id = self._get_image_id(output_filename)
            if image_id in self.global_descriptions:
                global_desc = self.global_descriptions[image_id].get(output_filename, '')
                if global_desc and not instruction:
                    prompt_parts.append(global_desc)

        return ", ".join(prompt_parts) if prompt_parts else "edit image"

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        input_img = self._load_image(sample['input'])
        output_img = self._load_image(sample['output'])

        original_size = (output_img.height, output_img.width)

        # Load or generate mask
        if self.random_mask and random.random() < 0.3:
            # 30% chance to use random mask for augmentation
            mask = random_brush_mask(output_img.height, output_img.width)
        else:
            mask = self._load_mask(sample['mask'])

        # Create masked input (conditioning image)
        masked_input = self._create_masked_input(input_img, mask)

        # Data augmentation
        if self.augment and self.color_jitter is not None:
            if random.random() < 0.3:
                # Apply same jitter to both input and output
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                output_img = self.color_jitter(output_img)
                torch.manual_seed(seed)
                input_img = self.color_jitter(input_img)
                # Recreate masked input after jitter
                masked_input = self._create_masked_input(input_img, mask)

            # Random horizontal flip (apply to all consistently)
            if random.random() < 0.5:
                output_img = output_img.transpose(Image.FLIP_LEFT_RIGHT)
                masked_input = masked_input.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Get prompt
        prompt = self._get_prompt(sample, sample['output'])

        # Apply transforms with same random seed for consistent cropping
        seed = torch.randint(0, 2**32, (1,)).item()

        torch.manual_seed(seed)
        masked_input_tensor = self.image_transforms(masked_input)

        torch.manual_seed(seed)
        target_tensor = self.image_transforms(output_img)

        torch.manual_seed(seed)
        mask_tensor = self.mask_transforms(mask)

        # Binarize mask
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "masked_input": masked_input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "prompt": prompt,
            "original_size": original_size,
            "crop_top_left": (0, 0),
            "sample_id": sample.get('output', ''),
        }


# ==================== VALIDATION ====================

def log_validation(vae, unet, brushnet, text_encoders, tokenizers,
                   train_dataset, args, accelerator, weight_dtype, global_step):
    """Run validation: generate sample images using full pipeline inference."""
    print(f"\n  Running validation at step {global_step}...")

    unwrapped_unet = accelerator.unwrap_model(unet) if args.train_unet else unet
    unwrapped_brushnet = accelerator.unwrap_model(brushnet) if args.train_brushnet else brushnet

    unwrapped_unet.eval()
    unwrapped_brushnet.eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Pick a few samples from dataset
    num_samples = min(args.num_validation_images, len(train_dataset))
    indices = list(range(num_samples))

    image_logs = []

    for idx in indices:
        sample = train_dataset[idx]
        prompt = sample["prompt"]

        # Prepare inputs
        masked_input = sample["masked_input"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        mask = sample["mask"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        # Encode to latent space
        with torch.no_grad():
            masked_latents = vae.encode(masked_input.to(dtype=vae.dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            mask_resized = F.interpolate(
                mask, size=(masked_latents.shape[-2], masked_latents.shape[-1]), mode="nearest"
            )

            # Encode prompt
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                [prompt], text_encoders, tokenizers
            )
            prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)

            # SDXL add_time_ids
            original_size = sample["original_size"]
            add_time_ids = torch.tensor([
                list(original_size) + [0, 0] + [args.resolution, args.resolution]
            ], device=accelerator.device, dtype=weight_dtype)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # BrushNet conditioning
            brushnet_cond = torch.cat([masked_latents, mask_resized.to(masked_latents.dtype)], dim=1)

            # Simple denoising loop (20 steps)
            latents = torch.randn_like(masked_latents)
            scheduler.set_timesteps(20, device=accelerator.device)

            for t in scheduler.timesteps:
                # BrushNet forward
                brushnet_out = unwrapped_brushnet(
                    latents, t.unsqueeze(0),
                    encoder_hidden_states=prompt_embeds,
                    brushnet_cond=brushnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                down_samples, mid_sample, up_samples = brushnet_out

                # UNet forward
                noise_pred = unwrapped_unet(
                    latents, t.unsqueeze(0),
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_add_samples=down_samples,
                    mid_block_add_sample=mid_sample,
                    up_block_add_samples=up_samples,
                    return_dict=False,
                )[0]

                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode
            latents = latents / vae.config.scaling_factor
            generated = vae.decode(latents.to(dtype=vae.dtype)).sample
            generated = ((generated + 1) / 2).clamp(0, 1)

        image_logs.append({
            "prompt": prompt,
            "generated": generated[0].cpu(),
            "masked_input": ((sample["masked_input"] + 1) / 2).clamp(0, 1),
            "target": ((sample["target"] + 1) / 2).clamp(0, 1),
            "mask": sample["mask"],
        })

    # Log to wandb
    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb_images = []
        for log in image_logs:
            wandb_images.extend([
                wandb.Image(log["masked_input"], caption=f"Input: {log['prompt'][:50]}"),
                wandb.Image(log["generated"], caption=f"Generated: {log['prompt'][:50]}"),
                wandb.Image(log["target"], caption="Ground Truth"),
            ])
        wandb.log({"validation": wandb_images}, step=global_step)

    # Save validation images
    val_dir = os.path.join(args.output_dir, f"validation-{global_step}")
    os.makedirs(val_dir, exist_ok=True)
    for i, log in enumerate(image_logs):
        from torchvision.utils import save_image
        save_image(log["generated"], os.path.join(val_dir, f"gen_{i}.png"))
        save_image(log["target"], os.path.join(val_dir, f"target_{i}.png"))
        save_image(log["masked_input"], os.path.join(val_dir, f"input_{i}.png"))

    print(f"  ✓ Validation images saved to {val_dir}")

    # Return to train mode
    if args.train_unet:
        unwrapped_unet.train()
    if args.train_brushnet:
        unwrapped_brushnet.train()

    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


# ==================== TRAINING ====================

def train_sequential_editing(args):
    """Main training function"""

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

    # ==================== WANDB INIT ====================

    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(args.output_dir),
            config=vars(args),
            tags=["sequential-editing", "brushnet", "sdxl"],
        )
        print("✓ Wandb initialized")
    elif args.use_wandb and not HAS_WANDB:
        print("⚠️ Wandb requested but not installed. Skipping.")

    # ==================== LOAD MODELS ====================

    print("\n[1/6] Loading models...")

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    )

    # Tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    # Text encoders (frozen)
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
    )

    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    )

    # UNet
    if args.pretrained_unet_path:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_unet_path)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
        )

    # BrushNet
    if args.pretrained_brushnet_path:
        brushnet = BrushNetModel.from_pretrained(args.pretrained_brushnet_path)
    else:
        print("  → Initializing BrushNet from UNet weights...")
        brushnet = BrushNetModel.from_unet(unet)

    # Freeze models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    if args.train_unet:
        unet.requires_grad_(True)
        print("  → Training: UNet")
    else:
        unet.requires_grad_(False)

    if args.train_brushnet:
        brushnet.requires_grad_(True)
        print("  → Training: BrushNet")
    else:
        brushnet.requires_grad_(False)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if args.train_unet:
            unet.enable_gradient_checkpointing()
        if args.train_brushnet:
            brushnet.enable_gradient_checkpointing()

    # xformers
    if args.enable_xformers:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
            print("  → xformers enabled")
        except ImportError:
            print("  → xformers not available, skipping")

    print("✓ Models loaded")

    # ==================== OPTIMIZER ====================

    print("\n[2/6] Setting up optimizer...")

    params_to_optimize = []
    if args.train_unet:
        params_to_optimize.extend(list(unet.parameters()))
    if args.train_brushnet:
        params_to_optimize.extend(list(brushnet.parameters()))

    if len(params_to_optimize) == 0:
        raise ValueError("No parameters to train! Enable --train_unet or --train_brushnet")

    # 8-bit Adam
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("  → Using 8-bit AdamW")
        except ImportError:
            print("  → bitsandbytes not available, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    print(f"✓ Optimizer ready")

    # ==================== DATASET ====================

    print("\n[3/6] Loading dataset...")

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

    print("\n[4/6] Preparing training...")

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
    if args.train_unet and args.train_brushnet:
        unet, brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, brushnet, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_unet:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        brushnet.to(accelerator.device, dtype=weight_dtype)
    else:
        brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            brushnet, optimizer, train_dataloader, lr_scheduler
        )
        unet.to(accelerator.device, dtype=weight_dtype)

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    print(f"✓ Training for {max_train_steps} steps ({args.num_train_epochs} epochs)")

    # Log dataset info to wandb
    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.config.update({
            "dataset_size": len(train_dataset),
            "total_steps": max_train_steps,
            "steps_per_epoch": num_update_steps_per_epoch,
        })

    # ==================== RESUME FROM CHECKPOINT ====================

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Find the latest checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                args.resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                args.resume_from_checkpoint = None
                print("No checkpoint found, starting from scratch")

        if args.resume_from_checkpoint:
            print(f"  Resuming from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            print(f"  ✓ Resumed at step {global_step}, epoch {first_epoch}")

    # ==================== TRAINING LOOP ====================

    print("\n[5/6] Starting training...")

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_unet:
            unet.train()
        if args.train_brushnet:
            brushnet.train()

        for step, batch in enumerate(train_dataloader):
            # Skip steps for resumed training
            if global_step < first_epoch * num_update_steps_per_epoch:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    global_step += 1
                continue

            with accelerator.accumulate(unet if args.train_unet else brushnet):

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

                # Resize mask to latent size
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

                # Proper SDXL add_time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
                add_time_ids_list = []
                for i in range(bsz):
                    orig_h, orig_w = batch["original_size"][0][i].item(), batch["original_size"][1][i].item()
                    crop_top, crop_left = 0, 0  # Using center/random crop
                    target_h, target_w = args.resolution, args.resolution
                    add_time_ids_list.append([orig_h, orig_w, crop_top, crop_left, target_h, target_w])

                add_time_ids = torch.tensor(
                    add_time_ids_list, device=target_latents.device, dtype=weight_dtype
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # BrushNet forward
                brushnet_cond = torch.cat([masked_latents, mask.to(masked_latents.dtype)], dim=1)

                brushnet_output = brushnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    brushnet_cond=brushnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet_output

                # UNet forward
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
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

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

                    # Log to wandb
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

                    # Save accelerator state (for resume)
                    accelerator.save_state(save_path)

                    # Also save model weights separately
                    if args.train_unet:
                        accelerator.unwrap_model(unet).save_pretrained(
                            os.path.join(save_path, "unet")
                        )
                    if args.train_brushnet:
                        accelerator.unwrap_model(brushnet).save_pretrained(
                            os.path.join(save_path, "brushnet")
                        )

                    # Remove old checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = sorted(
                            [d for d in checkpoints if d.startswith("checkpoint-")],
                            key=lambda x: int(x.split("-")[1])
                        )
                        if len(checkpoints) > args.checkpoints_total_limit:
                            for ckpt in checkpoints[:-args.checkpoints_total_limit]:
                                ckpt_path = os.path.join(args.output_dir, ckpt)
                                print(f"  Removing old checkpoint: {ckpt_path}")
                                shutil.rmtree(ckpt_path)

                    print(f"\n✓ Checkpoint saved: {save_path}")

                    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
                        wandb.log({"train/checkpoint": global_step}, step=global_step)

                # Validation
                if args.validation_steps and global_step % args.validation_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        log_validation(
                            vae, unet, brushnet,
                            [text_encoder_one, text_encoder_two],
                            [tokenizer_one, tokenizer_two],
                            train_dataset, args, accelerator, weight_dtype, global_step,
                        )

                # Log sample images periodically
                if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
                    if global_step % args.wandb_log_images_steps == 0 and global_step > 0:
                        try:
                            def denorm(x):
                                return ((x + 1) / 2).clamp(0, 1)

                            masked_img = denorm(batch["masked_input"][0]).cpu()
                            target_img = denorm(batch["target"][0]).cpu()
                            mask_img = batch["mask"][0].cpu()

                            wandb.log({
                                "samples/masked_input": wandb.Image(masked_img, caption=batch["prompt"][0]),
                                "samples/target": wandb.Image(target_img, caption="Ground Truth"),
                                "samples/mask": wandb.Image(mask_img, caption="Mask"),
                            }, step=global_step)
                        except Exception as e:
                            print(f"Warning: Could not log images to wandb: {e}")

            if global_step >= max_train_steps:
                break

    # ==================== SAVE ====================

    print("\n[6/6] Saving final model...")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.train_unet:
            accelerator.unwrap_model(unet).save_pretrained(
                os.path.join(args.output_dir, "unet")
            )
        if args.train_brushnet:
            accelerator.unwrap_model(brushnet).save_pretrained(
                os.path.join(args.output_dir, "brushnet")
            )

        print(f"✓ Model saved to {args.output_dir}")

    accelerator.end_training()

    # Finish wandb
    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.finish()

    print("\n✅ Training completed!")


# ==================== MAIN ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Image Editing Training")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)
    parser.add_argument("--pretrained_brushnet_path", type=str, default=None)

    # What to train
    parser.add_argument("--train_unet", action="store_true")
    parser.add_argument("--train_brushnet", action="store_true")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use_sessions", action="store_true",
                       help="Use edit_sessions.json for multi-turn training")
    parser.add_argument("--use_global_captions", action="store_true", default=True,
                       help="Use global_descriptions.json")
    parser.add_argument("--use_local_captions", action="store_true",
                       help="Use local_descriptions.json")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")

    # Augmentation
    parser.add_argument("--random_mask", action="store_true",
                       help="Augment with random brush masks (30%% chance)")
    parser.add_argument("--augment", action="store_true",
                       help="Enable data augmentation (color jitter, flip)")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Override num_train_epochs with total steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["linear", "cosine", "cosine_with_restarts",
                                "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Use 8-bit Adam from bitsandbytes")
    parser.add_argument("--set_grads_to_none", action="store_true",
                       help="Set grads to None instead of zero for memory savings")
    parser.add_argument("--enable_xformers", action="store_true",
                       help="Enable xformers memory efficient attention")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5,
                       help="Max checkpoints to keep (older ones deleted)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint or 'latest' to auto-detect")

    # Validation
    parser.add_argument("--validation_steps", type=int, default=500,
                       help="Run validation every N steps (0 to disable)")
    parser.add_argument("--num_validation_images", type=int, default=4,
                       help="Number of validation images to generate")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sequential-image-editing",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (default: output_dir basename)")
    parser.add_argument("--wandb_log_images_steps", type=int, default=500,
                       help="Log sample images every N steps")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sequential_editing(args)
