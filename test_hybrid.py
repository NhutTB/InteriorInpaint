"""
Test script for StableDiffusionXLHybridPipeline
Tests the hybrid pipeline with ControlNet + BrushNet + SDXL Inpainting
"""

import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add InteriorInpaint to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InteriorInpaint.pipelines import StableDiffusionXLHybridPipeline
from InteriorInpaint.models import BrushNetModel, UNet2DConditionModel

from diffusers import ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image


# ==================== CONFIGURATION ====================

# Base SDXL Model (can be DreamBooth fine-tuned)
BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
# BASE_MODEL_PATH = "path/to/your/dreamboothed/model"  # Uncomment after training

# BrushNet checkpoint
BRUSHNET_PATH = "data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v0"

# ControlNet checkpoint (MLSD for architectural lines)
CONTROLNET_PATH = "diffusers/controlnet-mlsd-sdxl-1.0"
# Or use depth: "diffusers/controlnet-depth-sdxl-1.0"

# Input images
IMAGE_PATH = "examples/test_image.jpg"  # Source image to inpaint
MASK_PATH = "examples/test_mask.jpg"   # Inpainting mask (white = inpaint region)
CONTROL_IMAGE_PATH = None  # Optional: pre-computed control image (e.g., MLSD edges)

# Prompt
PROMPT = "modern minimalist sofa with white cushions in a bright living room"
NEGATIVE_PROMPT = "ugly, blurry, low quality, distorted, artifacts"

# Generation settings
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
BRUSHNET_CONDITIONING_SCALE = 1.0
CONTROLNET_CONDITIONING_SCALE = 0.5  # Lower value for subtle structure guidance
CONTROL_GUIDANCE_START = 0.0
CONTROL_GUIDANCE_END = 1.0

SEED = 42
OUTPUT_PATH = "output_hybrid.png"

# ==================== HELPER FUNCTIONS ====================

def prepare_control_image_mlsd(image_path, output_size=(1024, 1024)):
    """Extract MLSD (line detection) from image for ControlNet"""
    try:
        from controlnet_aux import MLSDdetector
        mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        image = load_image(image_path)
        control_image = mlsd(image)
        return control_image
    except ImportError:
        print("⚠️ controlnet_aux not installed. Using original image as control.")
        print("Install with: pip install controlnet-aux")
        return load_image(image_path)

def prepare_control_image_depth(image_path, output_size=(1024, 1024)):
    """Extract depth map from image for ControlNet"""
    try:
        from transformers import pipeline
        depth_estimator = pipeline('depth-estimation')
        image = load_image(image_path)
        depth_map = depth_estimator(image)['depth']
        return depth_map
    except ImportError:
        print("⚠️ transformers not installed for depth. Using original image.")
        return load_image(image_path)

def load_and_preprocess_images(image_path, mask_path, target_size=1024):
    """Load and preprocess images to target size"""
    # Load images
    init_image = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to target size while maintaining aspect ratio
    h, w, _ = init_image.shape
    if w < h:
        scale = target_size / w
    else:
        scale = target_size / h
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    init_image = cv2.resize(init_image, (new_w, new_h))
    mask_image = cv2.resize(mask_image, (new_w, new_h))
    
    # Binarize mask (0 or 1)
    mask_image = (mask_image > 127).astype(np.uint8)
    
    # Create masked image (zero out inpainting region)
    masked_image = init_image * (1 - mask_image[:, :, np.newaxis])
    
    # Convert to PIL
    init_image_pil = Image.fromarray(init_image.astype(np.uint8))
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
    
    return init_image_pil, mask_image_pil, masked_image_pil

# ==================== MAIN TEST ====================

def test_hybrid_pipeline():
    print("=" * 60)
    print("Testing StableDiffusionXLHybridPipeline")
    print("=" * 60)
    
    # 1. Load models
    print("\n[1/6] Loading models...")
    
    # Load BrushNet
    print("  - Loading BrushNet...")
    brushnet = BrushNetModel.from_pretrained(
        BRUSHNET_PATH, 
        torch_dtype=torch.float16
    )
    
    # Load ControlNet
    print("  - Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=torch.float16
    )
    
    # Load VAE (use fp16-fix to avoid NaN)
    print("  - Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    
    # Load Hybrid Pipeline
    print("  - Loading Hybrid Pipeline...")
    pipe = StableDiffusionXLHybridPipeline.from_pretrained(
        BASE_MODEL_PATH,
        brushnet=brushnet,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    # Set scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU with CPU offloading
    pipe.enable_model_cpu_offload()
    print("  ✓ Models loaded successfully!")
    
    # 2. Prepare images
    print("\n[2/6] Preparing input images...")
    init_image, mask_image, masked_image = load_and_preprocess_images(
        IMAGE_PATH, MASK_PATH
    )
    print(f"  - Image size: {init_image.size}")
    
    # 3. Prepare ControlNet conditioning
    print("\n[3/6] Preparing ControlNet conditioning...")
    if CONTROL_IMAGE_PATH:
        control_image = load_image(CONTROL_IMAGE_PATH)
    else:
        # Auto-generate from source image
        if "mlsd" in CONTROLNET_PATH.lower():
            control_image = prepare_control_image_mlsd(IMAGE_PATH)
        elif "depth" in CONTROLNET_PATH.lower():
            control_image = prepare_control_image_depth(IMAGE_PATH)
        else:
            print("  ⚠️ Unknown ControlNet type. Using Canny edge detection...")
            # Simple Canny fallback
            img_np = np.array(init_image)
            edges = cv2.Canny(img_np, 100, 200)
            control_image = Image.fromarray(
                np.stack([edges, edges, edges], axis=-1)
            )
    
    control_image = control_image.resize(init_image.size)
    print("  ✓ Control image prepared!")
    
    # 4. Set seed for reproducibility
    print("\n[4/6] Setting random seed...")
    generator = torch.Generator("cuda").manual_seed(SEED)
    print(f"  - Seed: {SEED}")
    
    # 5. Run inference
    print("\n[5/6] Running hybrid inference...")
    print(f"  - Prompt: {PROMPT}")
    print(f"  - Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"  - Guidance scale: {GUIDANCE_SCALE}")
    print(f"  - BrushNet scale: {BRUSHNET_CONDITIONING_SCALE}")
    print(f"  - ControlNet scale: {CONTROLNET_CONDITIONING_SCALE}")
    
    output = pipe(
        prompt=PROMPT,
        prompt_2=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        negative_prompt_2=NEGATIVE_PROMPT,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        brushnet_image=masked_image,  # BrushNet uses masked image
        height=init_image.size[1],
        width=init_image.size[0],
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        brushnet_conditioning_scale=BRUSHNET_CONDITIONING_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        control_guidance_start=CONTROL_GUIDANCE_START,
        control_guidance_end=CONTROL_GUIDANCE_END,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )
    
    result_image = output.images[0]
    print("  ✓ Inference completed!")
    
    # 6. Save output
    print("\n[6/6] Saving output...")
    result_image.save(OUTPUT_PATH)
    print(f"  ✓ Saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)
    
    return result_image


if __name__ == "__main__":
    try:
        test_hybrid_pipeline()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
