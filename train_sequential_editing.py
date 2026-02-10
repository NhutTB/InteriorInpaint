"""
Training script for Sequential Image Editing with BrushNet + SDXL
Optimized for multi-turn instruction-based editing dataset

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
import json
import os
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

from diffusers import AutoencoderKL, DDPMScheduler
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
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.center_crop = center_crop
        self.use_global_captions = use_global_captions
        self.use_local_captions = use_local_captions
        
        # Load metadata
        self.images_dir = self.data_root / "images"
        
        # Load edit data
        if use_sessions:
            # Multi-turn mode: use sessions
            sessions_file = self.data_root / "edit_sessions.json"
            with open(sessions_file, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            # Flatten sessions into individual turns
            self.samples = []
            for session_id, turns in sessions.items():
                for turn in turns:
                    turn['session_id'] = session_id
                    self.samples.append(turn)
        else:
            # Single-turn mode: use edit_turns
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
        
        # Binarize mask (white = 255 = area to mask)
        mask_binary = (mask_np > 128).astype(np.uint8)
        
        # Black out masked region
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
        
        # Base instruction
        instruction = sample.get('instruction', '')
        if instruction:
            prompt_parts.append(instruction)
        
        # Add local description if available
        if self.use_local_captions:
            image_id = self._get_image_id(output_filename)
            if image_id in self.local_descriptions:
                local_desc = self.local_descriptions[image_id].get(output_filename, '')
                if local_desc:
                    prompt_parts.append(f"({local_desc})")
        
        # Add global description if available
        if self.use_global_captions:
            image_id = self._get_image_id(output_filename)
            if image_id in self.global_descriptions:
                global_desc = self.global_descriptions[image_id].get(output_filename, '')
                if global_desc and not instruction:
                    # Only use global if no instruction
                    prompt_parts.append(global_desc)
        
        return ", ".join(prompt_parts) if prompt_parts else "edit image"
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        input_img = self._load_image(sample['input'])
        output_img = self._load_image(sample['output'])
        mask = self._load_mask(sample['mask'])
        
        original_size = (output_img.height, output_img.width)
        
        # Create masked input (conditioning image)
        masked_input = self._create_masked_input(input_img, mask)
        
        # Get prompt
        prompt = self._get_prompt(sample, sample['output'])
        
        # Apply transforms with same random seed
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
            "sample_id": sample.get('output', ''),
        }


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
    
    optimizer = torch.optim.AdamW(
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
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # ==================== PREPARE ====================
    
    print("\n[4/6] Preparing training...")
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
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
        brushnet.to(accelerator.device)
    else:
        brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            brushnet, optimizer, train_dataloader, lr_scheduler
        )
        unet.to(accelerator.device)
    
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    
    print(f"✓ Training for {max_train_steps} steps ({args.num_train_epochs} epochs)")
    
    # Log dataset info to wandb
    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
        wandb.config.update({
            "dataset_size": len(train_dataset),
            "total_steps": max_train_steps,
            "steps_per_epoch": num_update_steps_per_epoch,
        })
    
    # ==================== TRAINING LOOP ====================
    
    print("\n[5/6] Starting training...")
    
    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_train_epochs):
        if args.train_unet:
            unet.train()
        if args.train_brushnet:
            brushnet.train()
        
        for step, batch in enumerate(train_dataloader):
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
                prompt_embeds_list = []
                
                for tokenizer, text_encoder in zip(
                    [tokenizer_one, tokenizer_two],
                    [text_encoder_one, text_encoder_two]
                ):
                    text_inputs = tokenizer(
                        batch["prompt"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    
                    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        output_hidden_states=True,
                    )
                    
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    prompt_embeds_list.append(prompt_embeds)
                
                encoder_hidden_states = torch.cat(prompt_embeds_list, dim=-1)
                
                # SDXL time embeddings
                add_time_ids = torch.cat([
                    torch.tensor(batch["original_size"][i]).unsqueeze(0)
                    for i in range(bsz)
                ], dim=0).to(target_latents.device)
                
                add_time_ids = torch.cat([
                    add_time_ids,
                    torch.zeros((bsz, 2), device=target_latents.device),
                    torch.tensor([[args.resolution, args.resolution]] * bsz, 
                                device=target_latents.device)
                ], dim=-1)
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(target_latents.device),
                    "time_ids": add_time_ids.to(target_latents.dtype),
                }
                
                # BrushNet forward
                brushnet_cond = torch.cat([masked_latents, mask.to(masked_latents.dtype)], dim=1)
                
                brushnet_output = brushnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    brushnet_cond=brushnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                
                down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet_output
                
                # UNet forward
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
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
                optimizer.zero_grad()
            
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
                    
                    if args.train_unet:
                        accelerator.unwrap_model(unet).save_pretrained(
                            os.path.join(save_path, "unet")
                        )
                    if args.train_brushnet:
                        accelerator.unwrap_model(brushnet).save_pretrained(
                            os.path.join(save_path, "brushnet")
                        )
                    
                    print(f"\n✓ Checkpoint saved: {save_path}")
                    
                    # Log checkpoint to wandb
                    if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
                        wandb.log({"train/checkpoint": global_step}, step=global_step)
                
                # Log sample images periodically
                if args.use_wandb and HAS_WANDB and accelerator.is_main_process:
                    if global_step % args.wandb_log_images_steps == 0 and global_step > 0:
                        # Log input/target/mask as images
                        try:
                            # Denormalize images
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
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Misc
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    
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
