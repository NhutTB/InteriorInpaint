# Hybrid Inpainting Pipeline - Architecture

## Overview

**InteriorInpaint** is a custom hybrid inpainting pipeline that combines three powerful techniques for high-quality interior design inpainting:

1. **ControlNet** - Structural guidance (preserves architectural elements)
2. **BrushNet** - Superior inpainting quality  
3. **SDXL** - Advanced image generation backbone

The pipeline allows fine-tuning via **DreamBooth** to adapt to specific interior styles.

---

## System Architecture

### Flow Diagram

```
INPUT IMAGE + MASK
        |
        ├─────────────────────────────────────────┐
        |                                         |
        v                                         v
   [ControlNet Path]                      [BrushNet Path]
        |                                         |
        v                                         v
Extract Structure Features                Create Latent Representation
(MLSD/Depth/Canny/Edge)                  (VAE encode + concat mask)
        |                                         |
        v                                         v
  ControlNet Forward                       BrushNet Forward
        |                                         |
        v                                         v
Down Residuals (12)                    Down Residuals (12)
Mid Residual (1)                       Mid Residual (1)
                                       Up Residuals (12)
        |                                         |
        └──────────────┬──────────────────────────┘
                       |
                       v
        ┌──────────────────────────────┐
        │  Text Encoder (CLIPTextModel) │
        │  (Prompt → Embedding)        │
        └──────────────┬───────────────┘
                       |
                       v
        ┌──────────────────────────────┐
        │  Modified UNet2DCondition    │
        │  - Accept ControlNet residuals
        │  - Accept BrushNet residuals │
        │  - Merged feature injection  │
        └──────────────┬───────────────┘
                       |
        ┌──────────────v───────────────┐
        │    Denoising Loop             │
        │    (Default: 50 Steps)        │
        │    - Noise prediction         │
        │    - Scheduler step           │
        └──────────────┬───────────────┘
                       |
                       v
        ┌──────────────────────────────┐
        │    VAE Decoder                │
        │    (Latent → Image space)     │
        └──────────────┬───────────────┘
                       |
                       v
              OUTPUT INPAINTED IMAGE
```

---

## Component Details

### 1. **ControlNet Branch** (Structural Guidance)

**File**: `models/brushnet.py` (ControlNet loaded from diffusers)

**Purpose**: Extract and preserve architectural structure

**Process**:
```
Original Image
    ↓
Feature Extraction (choose one):
    - MLSD: Line / edge detection
    - Canny: Edge detection  
    - Depth: Depth map estimation
    ↓
ControlNet Model (pretrained)
    ↓ 
Output:
    - 12 down block residuals
    - 1 mid block residual
```

**Key Point**: ControlNet learns to encode spatial structure independently, ensuring architectural consistency in the inpainted region.

---

### 2. **BrushNet Branch** (Inpainting Quality)

**File**: `models/brushnet.py` (BrushNetModel class)

**Purpose**: High-quality unmasked region inpainting with fine detail control

**Process**:
```
Masked Image (0 at mask region)
    ↓
VAE Encoder → Latent space (4D: B×C×H/8×W/8)
    ↓
Mask Resized to Latent Scale (B×1×H/8×W/8)
    ↓
Concatenate: [latent, mask] → (B×5×H/8×W/8)
    ↓
BrushNet Model
    ↓
Output:
    - 12 down block residuals
    - 1 mid block residual  
    - 12 up block residuals ← UNIQUE to BrushNet!
```

**Key Advantage**: BrushNet processes features at multiple scales (down, mid, **up**), enabling superior inpainting quality compared to ControlNet-only approaches.

---

### 3. **Modified UNet** (Residual Fusion)

**File**: `models/unets/unet_2d_condition.py`

**Purpose**: Accept and intelligently fuse residuals from both ControlNet and BrushNet

**Modifications from diffusers UNet**:

```python
# Forward signature now accepts:
- down_block_additional_residuals (ControlNet)
- mid_block_additional_residual (ControlNet)
- down_block_add_samples (BrushNet)        # NEW
- mid_block_add_sample (BrushNet)          # NEW
- up_block_add_samples (BrushNet)          # NEW
```

**Fusion Strategy** (in `forward()` method):

```python
# Down blocks (12 total)
for downsample_block in down_blocks:
    # Collect both ControlNet + BrushNet residuals
    additional_residuals = {}
    
    # Add ControlNet residuals if provided
    if down_block_additional_residuals is not None:
        additional_residuals["residuals"] = [
            down_block_additional_residuals[i] for i in ...
        ]
    
    # Add BrushNet residuals if provided  
    if down_block_add_samples is not None:
        additional_residuals["samples"] = [
            down_block_add_samples[i] for i in ...
        ]
    
    # Pass fused residuals to block
    hidden_states = downsample_block(
        hidden_states,
        temb,
        encoder_hidden_states,
        **additional_residuals
    )

# Mid block (1 total)
# Similar fusion strategy

# Up blocks (12 total)
# Similar fusion strategy + BrushNet up block residuals
```

#### Residual Injection Points

| Block Type | ControlNet | BrushNet | Fused |
|-----------|-----------|----------|-------|
| Down Block i | ✓ | ✓ | Concatenated |
| Mid Block | ✓ | ✓ | Concatenated |
| Up Block i | ✗ | ✓ | Only BrushNet |

---

### 4. **Hybrid Pipeline** (Orchestration)

**File**: `pipelines/pipeline_hybrid_sd_xl.py`

**Class**: `StableDiffusionXLHybridPipeline`

**Components Managed**:
- VAE (AutoencoderKL)
- Text Encoders (CLIP + CLIP-L)
- Tokenizers (CLIP)
- Custom UNet2DCondition
- BrushNet Model
- ControlNet Model(s)
- Scheduler (DDIMScheduler / PNDMScheduler)

**Main Method**: `__call__()`

```python
def __call__(
    self,
    prompt: str,
    image: PIL.Image,           # Original image
    mask: PIL.Image,            # Inpaint region
    control_image: PIL.Image,   # Structure guidance
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    strength: float = 1.0,      # 0→1: modify amount
    generator: Optional[torch.Generator] = None,
    ...
)
```

**Execution Steps**:

1. **Encode Prompt** → Embedding + pooled features
2. **Prepare Latents** → VAE encode + noise scaling
3. **Extract ControlNet Features** → Control image features
4. **Denoising Loop** (num_inference_steps):
   ```python
   for timestep in timesteps:
       # ControlNet forward
       down_block_res, mid_block_res = controlnet(
           latent_model_input,
           timestep,
           encoder_hidden_states,
           controlnet_cond=control_image_features,
       )
       
       # BrushNet forward
       up_block_res, down_block_res_bn, mid_block_res_bn = brushnet(
           latent_model_input,
           timestep,
           encoder_hidden_states,
           masked_latent=masked_latent,
       )
       
       # UNet forward (fused residuals)
       model_pred = unet(
           latent_model_input,
           timestep,
           encoder_hidden_states,
           down_block_additional_residuals=down_block_res,
           mid_block_additional_residual=mid_block_res,
           down_block_add_samples=down_block_res_bn,
           mid_block_add_sample=mid_block_res_bn,
           up_block_add_samples=up_block_res,
       )
       
       # Noise prediction + scheduler step
       latents = scheduler.step(model_pred, timestep, latents).prev_sample
   ```

5. **Decode Latents** → VAE decode to image space
6. **Return Output** → PIL Image

---

## Data Flow Summary

```
prompt, image, mask, control_image
    ↓
[Text Encoding] → prompt_embeds
[Image Encoding] → latents, masked_latent, control_features
    ↓
Denoising Loop:
    ├─ ControlNet(latents, control_features)
    │  ├─ down_block_residuals (12)
    │  └─ mid_block_residual (1)
    │
    ├─ BrushNet(masked_latent)
    │  ├─ down_block_residuals (12)
    │  ├─ mid_block_residual (1)
    │  └─ up_block_residuals (12)
    │
    └─ UNet(latents, residuals_fused)
       └─ noise_prediction
    ↓
VAE Decode → RGB Image
```

---

## Training Components

### DreamBooth Fine-tuning (Optional)

**File**: `train_dreambooth.py`

Train SDXL text encoder + UNet on interior design images to adapt to specific styles:

```python
# Minimal setup:
- 20-50 reference images
- 100 epochs  
- Learning rate: 1e-4
- Instance prompt: "a photo of [V] interior design"
```

### Model Variants

- **Base**: SDXL Base-1.0 (pretrained)
- **Fine-tuned**: DreamBooth-adapted SDXL
- **ControlNet**: Pretrained structural guidance
- **BrushNet**: Inpainting-specific weights

---

## Key Technical Innovations

1. **Dual Residual Injection**: ControlNet + BrushNet residuals fused systematically
2. **Up Block Support**: BrushNet up blocks for fine detail generation
3. **Mask-Aware Processing**: BrushNet concatenates mask in latent space
4. **Multi-Scale Guidance**: Structure (down channels) + Content (up channels)

---

## File Structure

```
models/
├── brushnet.py                      # BrushNet implementation
└── unets/
    └── unet_2d_condition.py         # Modified UNet (accepts dual residuals)

pipelines/
└── pipeline_hybrid_sd_xl.py         # Main hybrid pipeline orchestrator

training/
└── (DreamBooth training scripts)

test_hybrid.py                       # Interactive testing/inference
train_dreambooth.py                  # Fine-tuning script
```
