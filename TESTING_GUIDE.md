# 🧪 HƯỚNG DẪN TEST TRAINING PIPELINE

## Mục đích
Kiểm tra xem code có thể train được không, TRƯỚC KHI thực sự train model.

---

## ✅ CÁCH 1: Quick Validation (Khuyến nghị)

### Bước 1: Mở terminal/cmd tại thư mục InteriorInpaint
```bash
cd E:\final_project\Task-2\InteriorInpaint
```

### Bước 2: Chạy validation script
```bash
# Windows:
quick_validate.bat

# Hoặc trực tiếp:
python test_training_pipeline.py
```

### Bước 3: Xem kết quả

Script sẽ chạy **6 tests**:

1. ✅ **Imports** - Kiểm tra import modules
2. ✅ **Model Loading** - Khởi tạo models
3. ✅ **Forward Pass** - Test UNet forward với BrushNet/ControlNet residuals
4. ✅ **Gradient Flow** - Kiểm tra gradients flow đúng
5. ✅ **Training Loop** - Chạy 3 training steps thử
6. ✅ **Checkpoint Save/Load** - Test lưu và load checkpoint

### Kết quả mong đợi:
```
============================================================
TEST SUMMARY
============================================================
✅ PASS - Imports
✅ PASS - Model Loading
✅ PASS - Forward Pass
✅ PASS - Gradient Flow
✅ PASS - Training Loop
✅ PASS - Checkpoint Save/Load
============================================================
RESULT: 6/6 tests passed
============================================================

🎉 ALL TESTS PASSED - Training pipeline is ready!
```

---

## ✅ CÁCH 2: Test từng phần

### Test 1: Import modules
```bash
python test_import.py
```

**Mong đợi:**
```
Testing InteriorInpaint imports...
✓ UNet2DConditionModel imported
✓ BrushNetModel imported
✓ StableDiffusionXLHybridPipeline imported
All imports successful!
```

### Test 2: Forward pass với dummy data
```python
import torch
from InteriorInpaint.models.unets import UNet2DConditionModel

# Tạo UNet nhỏ để test
unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(32, 64),
)

# Test forward
sample = torch.randn(1, 4, 32, 32)
timestep = torch.tensor([1])
encoder_hidden_states = torch.randn(1, 10, 64)

output = unet(sample, timestep, encoder_hidden_states)
print(f"✅ Output shape: {output.sample.shape}")
```

### Test 3: Test với BrushNet residuals
```python
# Test với BrushNet residuals
down_add = [torch.randn(1, 4, 32, 32) for _ in range(3)]
mid_add = torch.randn(1, 4, 32, 32)
up_add = [torch.randn(1, 4, 32, 32) for _ in range(3)]

output = unet(
    sample, 
    timestep, 
    encoder_hidden_states,
    down_block_add_samples=down_add,
    mid_block_add_sample=mid_add,
    up_block_add_samples=up_add,
)
print(f"✅ BrushNet residuals test passed!")
```

### Test 4: Test gradient flow
```python
unet.train()
sample = torch.randn(1, 4, 32, 32, requires_grad=True)
output = unet(sample, timestep, encoder_hidden_states)

# Loss và backward
target = torch.randn_like(output.sample)
loss = torch.nn.functional.mse_loss(output.sample, target)
loss.backward()

# Check gradients
grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
print(f"✅ Parameters with gradients: {grad_count}")
```

---

## ⚠️ Nếu có lỗi

### Lỗi 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'diffusers'
```

**Giải pháp:**
```bash
pip install diffusers transformers accelerate torch torchvision
```

### Lỗi 2: Import InteriorInpaint failed
```
ModuleNotFoundError: No module named 'InteriorInpaint'
```

**Giải pháp:**
```bash
# Đảm bảo __init__.py files tồn tại
# Hoặc thêm vào path:
import sys
sys.path.insert(0, 'E:/final_project/Task-2/InteriorInpaint')
```

### Lỗi 3: CUDA out of memory (khi test)
```
torch.cuda.OutOfMemoryError
```

**Giải pháp:**
```python
# Test với CPU
device = "cpu"
unet = unet.to(device)
sample = sample.to(device)
```

### Lỗi 4: Gradient không flow
```
Parameters with gradients: 0/xxx
```

**Giải pháp:**
- Kiểm tra `unet.train()` đã được gọi chưa
- Kiểm tra `requires_grad=True` cho input
- Kiểm tra không có `.detach()` nào block gradient

---

## 📊 Checklist trước khi train thật

- [ ] ✅ Test imports passed
- [ ] ✅ Forward pass works
- [ ] ✅ BrushNet residuals integration works
- [ ] ✅ ControlNet residuals integration works
- [ ] ✅ Gradient flow correct
- [ ] ✅ Mini training loop runs
- [ ] ✅ Checkpoint save/load works
- [ ] 📂 Dataset prepared (20-50 images)
- [ ] 💾 Sufficient disk space (~10GB for checkpoints)
- [ ] 🎮 GPU available (16GB+ VRAM recommended)

---

## 🚀 Sau khi validation pass

### Option 1: Train ngay
```bash
run_training.bat
```

### Option 2: Tùy chỉnh parameters
```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="data/interior_images" \
  --instance_prompt="a photo of modern interior" \
  --output_dir="output/my_model" \
  --num_train_epochs=50 \
  --learning_rate=1e-6
```

---

## 💡 Tips

1. **Chạy validation TRƯỚC khi train**: Tiết kiệm thời gian debug
2. **Test với small model**: Fast iteration, catch errors early
3. **Check gradients**: Đảm bảo model học được
4. **Save checkpoints thường xuyên**: Tránh mất công training

---

## 📞 Troubleshooting

Nếu validation FAIL:
1. Đọc error message cẩn thận
2. Check log trong `test_training_pipeline.py`
3. Xem `VALIDATION_REPORT.md` cho known issues
4. Verify dependencies: `pip list | grep diffusers`

Nếu validation PASS nhưng training FAIL:
1. Kiểm tra dataset format
2. Kiểm tra VRAM usage
3. Thử giảm `batch_size` hoặc `resolution`
4. Enable `gradient_checkpointing`

---

**Tạo**: 2026-02-06  
**Script**: `test_training_pipeline.py`, `quick_validate.bat`
