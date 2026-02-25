import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import sys
import os
import time  # Thêm thư viện đo thời gian

# Import pipeline custom của bạn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines import StableDiffusionXLHybridPipeline
from models import BrushNetModel

# ---------------- CẤU HÌNH ----------------
IMAGE_PATH = "examples/test.jpg"   
MASK_PATH = "examples/mask.png"    
OUTPUT_PATH = "results/output_clock.png"

PROMPT = "A large high-quality luxury circular wall clock, black metal frame, white face with black numbers, realistic shadows and reflections, mounted centered on the white wall in a modern luxury living room with wooden furniture and natural light, 8k uhd, high detail, photorealistic"
NEGATIVE_PROMPT = "low quality, blurry, distorted, messy, watermark, text, deformed, floating, abstract, cartoon, empty wall, no clock, artifacts, mismatched lighting"

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
BRUSHNET_MODEL = "Brushnet-cpkt"  
# ---------------------------------------------

def load_and_prep_image(img_path, mask_path):
    init_img = Image.open(img_path).convert("RGB").resize((1024, 1024))
    
    # Hỗ trợ đọc cả mask trong suốt (Alpha channel)
    mask_raw = Image.open(mask_path)
    if mask_raw.mode in ('RGBA', 'LA') or (mask_raw.mode == 'P' and 'transparency' in mask_raw.info):
        mask_img = mask_raw.convert("RGBA").split()[-1]
    else:
        mask_img = mask_raw.convert("L")
        
    mask_img = mask_img.resize((1024, 1024))
    mask_np = np.array(mask_img)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    
    # Chống ngược Mask
    if np.mean(mask_np) > 127:
        mask_np = 255 - mask_np
        
    mask_img = Image.fromarray(mask_np)

    init_np = np.array(init_img)
    masked_np = init_np.copy()
    masked_np[mask_np == 255] = 0
    masked_img = Image.fromarray(masked_np)
    
    return init_img, mask_img, masked_img

def main():
    print("⏳ Đang khởi tạo hệ thống...")
    total_start_time = time.time() # Bắt đầu bấm giờ tổng
    
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
    
    # [NÂNG CẤP 1] - Đổi sang Karras Sigmas để ảnh mượt và thật hơn
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )
    
    pipe.enable_model_cpu_offload() 
    
    # [NÂNG CẤP 2] - Tối ưu VRAM bằng xformers (nếu máy có cài sẵn)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  - Đã bật Xformers giúp tăng tốc và giảm VRAM.")
    except Exception:
        pass

    init_image, mask_image, masked_image = load_and_prep_image(IMAGE_PATH, MASK_PATH)
    
    print("🎨 Bắt đầu quá trình Generate...")
    gen_start_time = time.time() # Bắt đầu bấm giờ lõi Generate
    generator = torch.Generator("cuda").manual_seed(42)

    output_images = pipe(
        prompt=PROMPT,
        prompt_2=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        negative_prompt_2=NEGATIVE_PROMPT,
        image=init_image,
        mask_image=mask_image,
        brushnet_image=masked_image,
        num_inference_steps=40,       # [NÂNG CẤP 3] - Tăng step lên 40 để lấy chi tiết cao hơn
        guidance_scale=7.5,
        brushnet_conditioning_scale=1.0,
        generator=generator,
        output_type="pil",
    ).images

    generated = output_images[0]
    gen_end_time = time.time()

    print("✂️ Đang xử lý hậu kỳ (Feathering Blend)...")
    # [NÂNG CẤP 4] - Mask Feathering: Làm mờ viền mask để ghép ảnh không bị lộ "vệt cắt"
    mask_np = np.array(mask_image)
    blurred_mask = cv2.GaussianBlur(mask_np, (15, 15), 0) # Làm mờ vùng biên
    
    init_np = np.array(init_image)
    gen_np = np.array(generated.resize(init_image.size))
    
    mask_norm = (blurred_mask / 255.0)[..., np.newaxis]
    composited = (gen_np * mask_norm + init_np * (1.0 - mask_norm)).astype(np.uint8)
    result = Image.fromarray(composited)

    os.makedirs("results", exist_ok=True)
    result.save(OUTPUT_PATH)
    total_end_time = time.time()
    
    print("\n" + "="*40)
    print(f"✅ HOÀN THÀNH TẠO ẢNH: {OUTPUT_PATH}")
    print(f"⏱️ Thời gian Generate: {gen_end_time - gen_start_time:.2f} giây")
    print(f"⏱️ Tổng thời gian (cả load model): {total_end_time - total_start_time:.2f} giây")
    print("="*40)

if __name__ == "__main__":
    main()