import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines import StableDiffusionXLHybridPipeline
from models import BrushNetModel

# ---------------- CẤU HÌNH ----------------
IMAGE_PATH = "examples/test.jpg"   
MASK_PATH = "examples/mask.png"    

PROMPT = "A modern living room with a round wall clock mounted on the white wall"
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text"

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
BRUSHNET_MODEL = "Brushnet-cpkt"  

# Thử nhiều scale để tìm giá trị tốt nhất
SCALES_TO_TEST = [0.3, 0.5, 0.7, 1.0]
# ---------------------------------------------

def load_and_prep_image(img_path, mask_path):
    init_image_original = Image.open(img_path).convert("RGB")
    
    mask_raw = Image.open(mask_path)
    if mask_raw.mode in ('RGBA', 'LA') or (mask_raw.mode == 'P' and 'transparency' in mask_raw.info):
        mask_gray = mask_raw.convert("RGBA").split()[-1]
    else:
        mask_gray = mask_raw.convert("L")
    
    target_size = (1024, 1024)
    init_np = np.array(init_image_original.resize(target_size))
    mask_np = np.array(mask_gray.resize(target_size))
    
    mask_binary = (mask_np > 127).astype(np.float64)
    if np.mean(mask_binary) > 0.5:
        mask_binary = 1.0 - mask_binary
    
    masked_np = (init_np * (1 - mask_binary[:, :, np.newaxis])).astype(np.uint8)
    masked_image = Image.fromarray(masked_np).convert("RGB")
    
    mask_rgb = Image.fromarray(
        (mask_binary.astype(np.uint8))[:, :, np.newaxis].repeat(3, axis=-1) * 255
    ).convert("RGB")
    
    return init_image_original, masked_image, mask_rgb, mask_binary


def main():
    print("⏳ Đang khởi tạo hệ thống...")
    total_start_time = time.time()
    
    from models.unets.unet_2d_condition import UNet2DConditionModel as CustomUNet
    custom_unet = CustomUNet.from_pretrained(SDXL_BASE, subfolder="unet", torch_dtype=torch.float16)
    brushnet = BrushNetModel.from_pretrained(BRUSHNET_MODEL, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    pipe = StableDiffusionXLHybridPipeline.from_pretrained(
        SDXL_BASE,
        unet=custom_unet,
        brushnet=brushnet,
        controlnet=None,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() 
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  - Đã bật Xformers")
    except Exception:
        pass

    init_image_original, masked_image, mask_rgb, mask_binary_np = load_and_prep_image(IMAGE_PATH, MASK_PATH)
    
    os.makedirs("results", exist_ok=True)

    # ===== THỬ NHIỀU SCALE =====
    for scale in SCALES_TO_TEST:
        print(f"\n🎨 Generate với brushnet_conditioning_scale={scale}...")
        gen_start = time.time()
        generator = torch.Generator("cuda").manual_seed(4321)

        output_images = pipe(
            prompt=PROMPT,
            prompt_2=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            negative_prompt_2=NEGATIVE_PROMPT,
            image=masked_image,
            mask_image=mask_rgb,
            brushnet_image=masked_image,
            num_inference_steps=50,
            guidance_scale=5.0,
            brushnet_conditioning_scale=scale,
            generator=generator,
            output_type="pil",
        ).images

        generated = output_images[0]
        gen_time = time.time() - gen_start
        
        # Lưu raw
        raw_path = f"results/clock_scale_{scale}_raw.png"
        generated.save(raw_path)
        
        # Composited
        generated_np = np.array(generated)
        init_np_original = np.array(init_image_original.resize(generated.size))
        mask_resized = cv2.resize(mask_binary_np, (generated.size[0], generated.size[1]))[:, :, np.newaxis]
        mask_blurred = cv2.GaussianBlur(mask_resized * 255, (21, 21), 0) / 255
        mask_blurred = mask_blurred[:, :, np.newaxis] if mask_blurred.ndim == 2 else mask_blurred
        mask_final = 1 - (1 - mask_resized) * (1 - mask_blurred)
        composited = (init_np_original * (1 - mask_final) + generated_np * mask_final).astype(np.uint8)
        
        comp_path = f"results/clock_scale_{scale}_comp.png"
        Image.fromarray(composited).save(comp_path)
        
        print(f"  ✅ Saved: {raw_path} | {comp_path} ({gen_time:.1f}s)")

    total_time = time.time() - total_start_time
    print(f"\n{'='*40}")
    print(f"✅ HOÀN THÀNH! Tổng thời gian: {total_time:.1f}s")
    print(f"📂 Kết quả trong: results/clock_scale_*.png")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()