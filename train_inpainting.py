"""
Training script for Supervised Inpainting with BrushNet + SDXL

Dataset structure:
    data/
    ├── inputs/           # Masked images (black region = inpaint area)
    │   ├── room1_masked.jpg
    │   └── ...
    ├── targets/          # Original images (ground truth)
    │   ├── room1.jpg
    │   └── ...
    ├── masks/            # Binary masks (optional)
    │   └── room1_mask.png
    └── metadata.jsonl    # Prompts
        {"input": "inputs/room1_masked.jpg", "target": "targets/room1.jpg", "prompt": "a red table lamp"}
        {"input": "inputs/room2_masked.jpg", "target": "targets/room2.jpg", "prompt": "a modern sofa"}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom models
from models.unets.unet_2d_condition import UNet2DConditionModel
from models.brushnet import BrushNetModel

check_min_version("0.27.0")


# ==================== DATASET ====================

class InpaintingDataset(Dataset):
    """
    Dataset for supervised inpainting training.
    
    Each sample contains:
    - masked_input: Image with black region (area to inpaint)
    - target: Original complete image (ground truth)  
    - mask: Binary mask (white = inpaint area)
    - prompt: Text description of what to inpaint
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.jsonl",
        size: int = 1024,
        center_crop: bool = False,
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.center_crop = center_crop
        
        # Load metadata
        self.samples = []
        metadata_path = self.data_root / metadata_file
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            # Auto-discover: match inputs/* with targets/*
            print(f"No {metadata_file} found, auto-discovering pairs...")
            inputs_dir = self.data_root / "inputs"
            targets_dir = self.data_root / "targets"
            
            if inputs_dir.exists() and targets_dir.exists():
                for input_path in inputs_dir.glob("*"):
                    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Find matching target
                        target_name = input_path.stem.replace("_masked", "")
                        for ext in ['.jpg', '.jpeg', '.png']:
                            target_path = targets_dir / f"{target_name}{ext}"
                            if target_path.exists():
                                self.samples.append({
                                    "input": str(input_path.relative_to(self.data_root)),
                                    "target": str(target_path.relative_to(self.data_root)),
                                    "prompt": "interior object"  # Default prompt
                                })
                                break
        
        if len(self.samples) == 0:
            raise ValueError(f"No training pairs found in {data_root}")
        
        print(f"Found {len(self.samples)} training pairs")
        
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
    
    def _extract_mask_from_masked_image(self, masked_img, target_img):
        """Extract mask by comparing masked input with target"""
        import numpy as np
        
        masked_np = np.array(masked_img)
        target_np = np.array(target_img)
        
        # Mask is where images differ significantly (or where input is black)
        diff = np.abs(masked_np.astype(float) - target_np.astype(float))
        mask = (diff.mean(axis=-1) > 30) | (masked_np.mean(axis=-1) < 10)
        
        return Image.fromarray((mask * 255).astype(np.uint8))
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        input_path = self.data_root / sample["input"]
        target_path = self.data_root / sample["target"]
        
        masked_input = Image.open(input_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        
        original_size = (target.height, target.width)
        
        # Load or extract mask
        if "mask" in sample and sample["mask"]:
            mask_path = self.data_root / sample["mask"]
            mask = Image.open(mask_path).convert("L")
        else:
            # Extract mask from difference
            mask = self._extract_mask_from_masked_image(masked_input, target)
        
        # Apply transforms
        # Use same random crop for all
        seed = torch.randint(0, 2**32, (1,)).item()
        
        torch.manual_seed(seed)
        masked_input = self.image_transforms(masked_input)
        
        torch.manual_seed(seed)
        target = self.image_transforms(target)
        
        torch.manual_seed(seed)
        mask = self.mask_transforms(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return {
            "masked_input": masked_input,
            "target": target,
            "mask": mask,
            "prompt": sample.get("prompt", "interior object"),
            "original_size": original_size,
        }


# ==================== TRAINING ====================

def train_inpainting(args):
    """Main training function for supervised inpainting"""
    
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
        # Initialize from UNet if no pretrained
        print("  → Initializing BrushNet from UNet weights...")
        brushnet = BrushNetModel.from_unet(unet)
    
    # Freeze models based on args
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
    
    print(f"✓ Optimizer ready ({len(params_to_optimize)} parameter groups)")
    
    # ==================== DATASET ====================
    
    print("\n[3/6] Loading dataset...")
    
    train_dataset = InpaintingDataset(
        data_root=args.data_dir,
        metadata_file=args.metadata_file,
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
                
                # ===== 1. Encode images to latent space =====
                with torch.no_grad():
                    # Encode target (ground truth)
                    target_latents = vae.encode(
                        batch["target"].to(dtype=vae.dtype)
                    ).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor
                    
                    # Encode masked input (for BrushNet conditioning)
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
                
                # ===== 2. Add noise to TARGET latents =====
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=target_latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # ===== 3. Encode prompts =====
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
                
                # ===== 4. Prepare SDXL time embeddings =====
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
                
                # ===== 5. BrushNet forward =====
                # BrushNet conditioning: masked_latents + mask
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
                
                # ===== 6. UNet forward with BrushNet residuals =====
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
                
                # ===== 7. Compute loss =====
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # ===== 8. Backward =====
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Progress update
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
                
                # Save checkpoint
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
                    
                    print(f"\n✓ Saved checkpoint to {save_path}")
            
            if global_step >= max_train_steps:
                break
    
    # ==================== SAVE FINAL ====================
    
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
    print("\n✅ Training completed!")


# ==================== MAIN ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Inpainting Training")
    
    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to pretrained SDXL model")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None,
                       help="Path to fp16-fix VAE")
    parser.add_argument("--pretrained_unet_path", type=str, default=None,
                       help="Path to pretrained UNet (optional)")
    parser.add_argument("--pretrained_brushnet_path", type=str, default=None,
                       help="Path to pretrained BrushNet (optional)")
    
    # What to train
    parser.add_argument("--train_unet", action="store_true",
                       help="Train UNet")
    parser.add_argument("--train_brushnet", action="store_true",
                       help="Train BrushNet")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--metadata_file", type=str, default="metadata.jsonl",
                       help="Metadata file name")
    parser.add_argument("--resolution", type=int, default=1024,
                       help="Training resolution")
    parser.add_argument("--center_crop", action="store_true",
                       help="Center crop images")
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Misc
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_inpainting(args)
