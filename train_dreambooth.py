"""
Training script for Interior Inpainting with DreamBooth SDXL
Trains SDXL base model on interior design dataset for style fine-tuning

Dataset structure:
    train/
        image_001.jpg
        image_002.jpg
        ...
    (Optional) metadata.jsonl - for custom prompts per image
"""

import argparse
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
from transformers import AutoTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version

# Check minimum version
check_min_version("0.27.0")

# ==================== DATASET ====================

class InteriorDataset(Dataset):
    """Simple interior design dataset for DreamBooth training"""
    
    def __init__(
        self,
        data_root,
        instance_prompt,
        size=1024,
        center_crop=False,
    ):
        self.data_root = Path(data_root)
        self.instance_prompt = instance_prompt
        self.size = size
        self.center_crop = center_crop
        
        # Load all images
        self.image_paths = list(self.data_root.glob("*.jpg")) + \
                          list(self.data_root.glob("*.png")) + \
                          list(self.data_root.glob("*.jpeg"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root}")
        
        print(f"Found {len(self.image_paths)} training images")
        
        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        original_size = (image.height, image.width)
        image = self.transforms(image)
        
        return {
            "pixel_values": image,
            "prompt": self.instance_prompt,
            "original_size": original_size,
        }

# ==================== TRAINING FUNCTION ====================

def train_dreambooth(args):
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
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== LOAD MODELS ====================
    
    print("\n[1/5] Loading models...")
    
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
    
    # Text encoders
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
    )
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    )
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    
    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # Enable UNet gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    print("✓ Models loaded")
    
    # ==================== SETUP OPTIMIZER ====================
    
    print("\n[2/5] Setting up optimizer...")
    
    params_to_optimize = unet.parameters()
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    print("✓ Optimizer ready")
    
    # ==================== DATASET ====================
    
    print("\n[3/5] Loading dataset...")
    
    train_dataset = InteriorDataset(
        data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # ==================== PREPARE FOR TRAINING ====================
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    
    print(f"✓ Training for {max_train_steps} steps ({args.num_train_epochs} epochs)")
    
    # ==================== TRAINING LOOP ====================
    
    print("\n[4/5] Starting training...")
    
    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
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
                    
                    # Use penultimate layer
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    
                    prompt_embeds_list.append(prompt_embeds)
                
                # Concatenate embeddings
                encoder_hidden_states = torch.cat(prompt_embeds_list, dim=-1)
                
                # Prepare added_cond_kwargs for SDXL
                add_time_ids = torch.cat([
                    torch.tensor(batch["original_size"][i]).repeat(1, 1)
                    for i in range(bsz)
                ]).to(latents.device)
                
                add_time_ids = torch.cat([
                    add_time_ids,
                    torch.zeros((bsz, 2), device=latents.device),  # crop coords
                    torch.tensor([[args.resolution, args.resolution]] * bsz, device=latents.device)  # target size
                ], dim=-1)
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                # Compute loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    print(f"✓ Saved checkpoint to {save_path}")
            
            if global_step >= max_train_steps:
                break
    
    # ==================== SAVE FINAL MODEL ====================
    
    print("\n[5/5] Saving final model...")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
        )
        
        pipeline.save_pretrained(args.output_dir)
        print(f"✓ Model saved to {args.output_dir}")
    
    accelerator.end_training()
    print("\n✅ Training completed!")

# ==================== MAIN ====================

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to pretrained SDXL model")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None,
                       help="Path to fp16-fix VAE (recommended: madebyollin/sdxl-vae-fp16-fix)")
    
    # Data
    parser.add_argument("--instance_data_dir", type=str, required=True,
                       help="Path to training images")
    parser.add_argument("--instance_prompt", type=str, required=True,
                       help="Instance prompt (e.g., 'a photo of modern interior design')")
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
    parser.add_argument("--learning_rate", type=float, default=1e-6,
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
    train_dreambooth(args)
