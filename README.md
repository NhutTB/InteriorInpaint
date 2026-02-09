# InteriorInpaint - Hybrid SDXL Inpainting Pipeline

Custom SDXL Inpainting pipeline combining **DreamBooth + ControlNet + BrushNet** for high-quality interior design inpainting.

## 📂 Project Structure

```
InteriorInpaint/
├── models/
│   ├── brushnet.py          # BrushNet model (from BrushNet repo)
│   └── unets/
│       └── unet_2d_condition.py  # Modified UNet (supports both ControlNet + BrushNet)
├── pipelines/
│   └── pipeline_hybrid_sd_xl.py  # Hybrid pipeline (NEW - combines all components)
├── training/
│   ├── train_brushnet_sdxl.py    # BrushNet training (from BrushNet repo)
│   └── train_dreambooth_lora_sdxl.py  # DreamBooth LoRA training (from diffusers)
├── train_dreambooth.py       # Simplified training script
├── test_hybrid.py            # Test/inference script
├── test_import.py            # Import verification
├── CODE_REVIEW.md            # Code review report
└── VALIDATION_REPORT.md      # Validation findings
```

## ✨ Features

- **Hybrid Architecture**: Combines 3 powerful techniques
  - **DreamBooth**: Fine-tune SDXL on your interior style
  - **ControlNet**: Structural guidance (MLSD/Depth)
  - **BrushNet**: Superior inpainting quality
  
- **Custom UNet**: Modified to accept residuals from both ControlNet AND BrushNet simultaneously

- **Ready-to-use Scripts**: Training and testing scripts included

## 🚀 Quick Start

### 1. Installation

```bash
pip install diffusers transformers accelerate torch torchvision
pip install controlnet-aux  # For control image preprocessing (optional)
```

### 2. Test the Pipeline

```bash
python test_hybrid.py \
  --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
  --brushnet_path "path/to/brushnet/checkpoint" \
  --controlnet_path "diffusers/controlnet-mlsd-sdxl-1.0" \
  --image "examples/test_image.jpg" \
  --mask "examples/test_mask.jpg" \
  --prompt "modern minimalist sofa with white cushions"
```

### 3. Train DreamBooth (Fine-tune on your style)

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="data/interior_images/" \
  --instance_prompt="a photo of modern interior design" \
  --output_dir="output/dreambooth_interior" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-6 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --mixed_precision="fp16"
```

## 📋 Dataset Structure

### For DreamBooth Training:
```
data/interior_images/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
...
```

- **20-50 images** recommended
- All should be similar **interior style** (e.g., modern minimalist, luxury classic, etc.)
- Resolution: **1024x1024** or higher

## 🔧 Key Files Explained

### `models/unets/unet_2d_condition.py`
**Modified UNet** that accepts:
- `down_block_additional_residuals` (ControlNet)
- `down_block_add_samples` (BrushNet down blocks)
- `mid_block_add_sample` (BrushNet mid block)
- `up_block_add_samples` (BrushNet up blocks)

**Critical fix applied**: BrushNet residuals are now passed INTO blocks (not added after), matching BrushNet's original design.

### `pipelines/pipeline_hybrid_sd_xl.py`
**New Hybrid Pipeline** that:
1. Loads SDXL + ControlNet + BrushNet
2. Runs ControlNet on structural image (edges/depth)
3. Runs BrushNet on masked image
4. Passes BOTH residuals to custom UNet
5. Produces high-quality inpainted result

### `test_hybrid.py`
Complete inference script with:
- Auto control image generation (MLSD/Depth)
- Image preprocessing
- Blended output option

## ⚙️ Configuration Examples

### For MLSD ControlNet (Architectural lines):
```python
CONTROLNET_PATH = "diffusers/controlnet-mlsd-sdxl-1.0"
CONTROLNET_CONDITIONING_SCALE = 0.5
```

### For Depth ControlNet:
```python
CONTROLNET_PATH = "diffusers/controlnet-depth-sdxl-1.0"  
CONTROLNET_CONDITIONING_SCALE = 0.3
```

### BrushNet Settings:
```python
BRUSHNET_CONDITIONING_SCALE = 1.0  # Full strength for best inpainting
```

## 🐛 Known Issues & Fixes

### ✅ SOLVED: Down Blocks Logic Error
**Issue**: Initial implementation added BrushNet residuals AFTER block execution.  
**Fix**: Now passes residuals INTO blocks as `**additional_residuals`.  
**Impact**: BrushNet now works correctly.

See `VALIDATION_REPORT.md` for details.

## 📊 Training Tips

1. **DreamBooth Learning Rate**: Start with `1e-6`, increase to `5e-6` if underfitting
2. **Batch Size**: Use `1` with `gradient_accumulation_steps=4` for 16GB VRAM
3. **Epochs**: 50-100 for small datasets (20-30 images), 30-50 for larger (50+ images)
4. **VAE**: Always use `madebyollin/sdxl-vae-fp16-fix` to avoid NaN issues

## 🎯 Use Cases

### Interior Design:
- Replace furniture while keeping room structure
- Change wall colors/materials  
- Add/remove decorative elements

### Architectural Visualization:
- Modify building facades
- Change interior layouts
- Style transfer (modern → classic, etc.)

## 📝 Citation

This project combines code and concepts from:
- [BrushNet](https://github.com/TencentARC/BrushNet)
- [Diffusers](https://github.com/huggingface/diffusers)
- [DreamBooth](https://dreambooth.github.io/)

## 📄 License

See individual component licenses:
- BrushNet: Apache 2.0
- Diffusers: Apache 2.0
- SDXL: CreativeML Open RAIL++-M

## 🤝 Contributing

This is a research/experimental project. Contributions welcome!

---

**Created**: 2026-02-06  
**Status**: ✅ Core implementation complete, ready for testing
