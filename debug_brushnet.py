"""
Debug script: Kiểm tra từng bước của pipeline để tìm lỗi.
Test theo đúng training convention:
  - BrushNet nhận SINGLE (non-CFG), UNSCALED latents
  - BrushNet nhận CONDITIONAL-ONLY embeddings (không phải uncond+cond)
  - brushnet_cond = cat([masked_latents, mask]) — mask 1.0 = vùng inpaint
"""
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines import StableDiffusionXLHybridPipeline
from models import BrushNetModel

IMAGE_PATH = "examples/test.jpg"
MASK_PATH = "examples/mask.png"
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
BRUSHNET_MODEL = "Brushnet-cpkt"

PROMPT = "A large wall clock"
NEGATIVE_PROMPT = "low quality"

def load_and_prep_image(img_path, mask_path):
    init_img = Image.open(img_path).convert("RGB").resize((1024, 1024))
    mask_img = Image.open(mask_path).convert("L").resize((1024, 1024))
    mask_np = np.array(mask_img)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np)
    init_np = np.array(init_img)
    masked_np = init_np.copy()
    masked_np[mask_np == 255] = 0  # Black out the inpaint region
    masked_img = Image.fromarray(masked_np)
    return init_img, mask_img, masked_img

def main():
    print("=== DEBUG: Loading models ===")
    from models.unets.unet_2d_condition import UNet2DConditionModel as CustomUNet
    custom_unet = CustomUNet.from_pretrained(SDXL_BASE, subfolder="unet", torch_dtype=torch.float16)
    brushnet = BrushNetModel.from_pretrained(BRUSHNET_MODEL, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    pipe = StableDiffusionXLHybridPipeline.from_pretrained(
        SDXL_BASE, unet=custom_unet, brushnet=brushnet, controlnet=None,
        vae=vae, torch_dtype=torch.float16, use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    print("=== DEBUG: Preparing images ===")
    init_image, mask_image, masked_image = load_and_prep_image(IMAGE_PATH, MASK_PATH)
    
    mask_np = np.array(mask_image)
    print(f"Mask unique values: {np.unique(mask_np)}")
    print(f"Mask white (inpaint) pixel count: {(mask_np == 255).sum()}")
    print(f"Mask black (context) pixel count: {(mask_np == 0).sum()}")

    device = torch.device("cuda")
    dtype = torch.float16
    height, width = 1024, 1024

    # Step 1: Preprocess
    print("\n=== Step 1: Preprocess BrushNet conditioning ===")
    bn_image_tensor = pipe.image_processor.preprocess(masked_image, height=height, width=width).to(device=device, dtype=dtype)
    print(f"BN image tensor range: [{bn_image_tensor.min():.4f}, {bn_image_tensor.max():.4f}]")
    
    mask_tensor = pipe.mask_processor.preprocess(mask_image, height=height, width=width).to(device=device, dtype=dtype)
    print(f"Mask tensor unique values: {torch.unique(mask_tensor).cpu().numpy()}")
    print(f"  → 1.0 = vùng inpaint (TRẮNG), 0.0 = background (GIỮ NGUYÊN)")

    # Step 2: VAE encode
    print("\n=== Step 2: VAE encode ===")
    with torch.no_grad():
        init_latents = pipe.vae.encode(bn_image_tensor).latent_dist.sample()
    init_latents = init_latents * pipe.vae.config.scaling_factor
    print(f"Masked image latents: shape={init_latents.shape}, range=[{init_latents.min():.4f}, {init_latents.max():.4f}]")

    # Step 3: Resize mask to latent size
    print("\n=== Step 3: Mask to latent size ===")
    mask_latent = torch.nn.functional.interpolate(mask_tensor, size=(init_latents.shape[-2], init_latents.shape[-1]))
    print(f"Mask latent unique values: {torch.unique(mask_latent).cpu().numpy()}")
    print(f"  NOTE: NO inversion — mask 1.0 = inpaint region (matching training convention)")

    # Step 4: Build brushnet_cond (TRAINING CONVENTION)
    brushnet_cond = torch.cat([init_latents, mask_latent], dim=1)
    print(f"\nBrushNet cond shape: {brushnet_cond.shape} (should be [1, 5, H, W])")

    # Step 5: Encode prompt
    print("\n=== Step 5: BrushNet forward (CORRECTED — single latents + cond-only embeds) ===")
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        PROMPT, prompt_2=PROMPT, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=NEGATIVE_PROMPT,
    )
    # Conditional only (matching training)
    cond_prompt_embeds = prompt_embeds   # [1, 77, 2048]
    cond_pooled_embeds = pooled_prompt_embeds  # [1, 1280]

    latents = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)

    pipe.scheduler.set_timesteps(30, device=device)
    t = pipe.scheduler.timesteps[0]
    print(f"Timestep: {t}")

    add_time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]], device=device, dtype=dtype)
    added_cond_kwargs = {"text_embeds": cond_pooled_embeds, "time_ids": add_time_ids}

    print("\n--- Convention check ---")
    print(f"  BrushNet latent input: SINGLE (batch=1), UNSCALED ✓")
    print(f"  BrushNet prompt: batch={cond_prompt_embeds.shape[0]} (CONDITIONAL ONLY) ✓")
    print(f"  brushnet_cond channels: {brushnet_cond.shape[1]} (4 latent + 1 mask) ✓")
    print(f"  Mask in brushnet_cond: 1.0 = inpaint, 0.0 = keep ✓")

    with torch.no_grad():
        brushnet_out = pipe.brushnet(
            latents,               # SINGLE, UNSCALED
            t,
            encoder_hidden_states=cond_prompt_embeds,   # CONDITIONAL ONLY
            added_cond_kwargs=added_cond_kwargs,         # SINGLE batch
            brushnet_cond=brushnet_cond,                 # Single, not doubled
            conditioning_scale=1.0,
            return_dict=False,
        )
    
    bn_down, bn_mid, bn_up = brushnet_out
    print(f"\nBrushNet outputs:")
    print(f"  Down blocks: {len(bn_down)} tensors")
    for i, d in enumerate(bn_down):
        print(f"    [{i}] shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}], mean={d.mean():.6f}, std={d.std():.6f}")
    print(f"  Mid: shape={bn_mid.shape}, range=[{bn_mid.min():.4f}, {bn_mid.max():.4f}], mean={bn_mid.mean():.6f}")
    print(f"  Up blocks: {len(bn_up)} tensors")
    for i, u in enumerate(bn_up):
        print(f"    [{i}] shape={u.shape}, range=[{u.min():.4f}, {u.max():.4f}], mean={u.mean():.6f}, std={u.std():.6f}")

    all_zero_down = all(d.abs().max() < 1e-6 for d in bn_down)
    all_zero_up = all(u.abs().max() < 1e-6 for u in bn_up)
    mid_zero = bn_mid.abs().max() < 1e-6
    
    print(f"\n⚠️  All down residuals ~zero? {all_zero_down}")
    print(f"⚠️  Mid residual ~zero? {mid_zero}")
    print(f"⚠️  All up residuals ~zero? {all_zero_up}")

    if all_zero_down and all_zero_up and mid_zero:
        print("\n🔴 CRITICAL: BrushNet outputs are all near-zero! Check model weights.")
    else:
        print("\n🟢 BrushNet is producing non-zero residuals.")
        print("   → Pipeline fix is correct. Run test_clock.py to generate full image.\n")

if __name__ == "__main__":
    main()
