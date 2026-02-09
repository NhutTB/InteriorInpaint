# 🎯 START HERE - InteriorInpaint Project

**Bắt đầu từ đây nếu bạn mới clone/download project này!**

---

## 📋 TÓM TẮT PROJECT

**InteriorInpaint** là custom SDXL Inpainting pipeline kết hợp:
- ✨ **DreamBooth** - Fine-tune style riêng
- 🎨 **BrushNet** - Inpainting chất lượng cao
- 🏗️ **ControlNet** - Kiểm soát cấu trúc

→ **Mục đích**: Inpainting nội thất với chất lượng cao và kiểm soát tốt

---

## 🚀 QUICK START (3 bước)

### Bước 1️⃣: Cài đặt môi trường

```bash
cd E:\final_project\Task-2\InteriorInpaint

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài đặt dependencies
install.bat
```

**Hoặc xem**: `INSTALL_QUICK.md`

---

### Bước 2️⃣: Test xem code hoạt động

```bash
# Test imports
python test_import.py

# Test training pipeline (6 tests)
python test_training_pipeline.py
```

**Kết quả mong đợi**: `6/6 tests passed ✅`

**Nếu có lỗi**: Xem `TESTING_GUIDE.md`

---

### Bước 3️⃣: Chọn workflow

#### A. Chỉ muốn TEST (không train):
```bash
# Cần: BrushNet checkpoint (download từ BrushNet repo)
python test_hybrid.py
```

#### B. Muốn TRAIN model riêng:
```bash
# 1. Chuẩn bị dataset (20-50 ảnh interior)
mkdir data\interior_images
# Copy ảnh vào đây

# 2. Train
run_training.bat

# 3. Test với model đã train
python test_hybrid.py --base_model="output/dreambooth_interior"
```

---

## 📂 CẤU TRÚC PROJECT

```
InteriorInpaint/
│
├── 📘 START_HERE.md              ← BẠN ĐANG Ở ĐÂY
├── 📘 README.md                  ← Overview & features
├── 📘 PROJECT_COMPLETE.md        ← Detailed guide
│
├── 📦 INSTALLATION
│   ├── INSTALL_QUICK.md          ← Quick install (3 lệnh)
│   ├── INSTALLATION.md           ← Chi tiết cài đặt
│   ├── requirements.txt          ← Dependencies chính
│   ├── requirements-optional.txt ← Optional packages
│   └── install.bat               ← Auto install script
│
├── 🧪 TESTING
│   ├── QUICK_TEST_GUIDE.md       ← Quick test guide
│   ├── TESTING_GUIDE.md          ← Detailed testing
│   ├── test_import.py            ← Test imports
│   ├── test_training_pipeline.py ← Test training (6 tests)
│   ├── quick_validate.bat        ← Quick validation
│   └── test_hybrid.py            ← Test inference
│
├── 🏋️ TRAINING
│   ├── train_dreambooth.py       ← Training script
│   └── run_training.bat          ← Quick train
│
├── 📖 DOCUMENTATION
│   ├── ARCHITECTURE.md           ← Kiến trúc technical
│   ├── CODE_REVIEW.md            ← So sánh với mã gốc
│   └── VALIDATION_REPORT.md      ← Lỗi đã sửa
│
└── 🔧 SOURCE CODE
    ├── models/                   ← Modified UNet, BrushNet
    ├── pipelines/                ← Hybrid pipeline
    └── __init__.py
```

---

## 🎯 WORKFLOW ĐỀ XUẤT

### Lần đầu sử dụng:

```
1. CÀI ĐẶT
   ↓
   install.bat
   ↓
2. VALIDATE
   ↓
   python test_training_pipeline.py
   ↓
   ✅ 6/6 PASS?
   ↓
3. QUYẾT ĐỊNH:
   
   A) CHỈ TEST         B) TRAIN MODEL
      ↓                    ↓
   Download BrushNet   Chuẩn bị dataset
      ↓                    ↓
   test_hybrid.py      run_training.bat
                           ↓
                       test_hybrid.py
```

---

## 📚 HƯỚNG DẪN CHI TIẾT

### Cài đặt:
- 🚀 **Quick**: `INSTALL_QUICK.md` (3 lệnh)
- 📖 **Chi tiết**: `INSTALLATION.md` (troubleshooting, platform-specific)

### Testing:
- 🚀 **Quick**: `QUICK_TEST_GUIDE.md`
- 📖 **Chi tiết**: `TESTING_GUIDE.md`

### Training:
- 📖 **Project overview**: `README.md`
- 📖 **Complete guide**: `PROJECT_COMPLETE.md`

### Technical:
- 🏗️ **Architecture**: `ARCHITECTURE.md`
- 🔍 **Code review**: `CODE_REVIEW.md`
- 🐛 **Bug fixes**: `VALIDATION_REPORT.md`

---

## ⚡ COMMANDS CHEAT SHEET

```bash
# INSTALLATION
install.bat                        # Auto install all

# TESTING
python test_import.py              # Test imports only
python test_training_pipeline.py  # Full validation (6 tests)
quick_validate.bat                 # Quick validation

# TRAINING
run_training.bat                   # Train with defaults
python train_dreambooth.py --help  # See all options

# INFERENCE
quick_test.bat                     # Quick inference test
python test_hybrid.py              # Full inference
```

---

## 🔧 DEPENDENCIES

### Core (Bắt buộc):
```
torch>=2.0.0
diffusers>=0.27.0
transformers>=4.35.0
accelerate>=0.25.0
```

### Optional (Khuyến nghị):
```
xformers       # 2-3x faster training
controlnet-aux # Auto control image generation
```

**Install**: `pip install -r requirements.txt`

---

## 💾 SYSTEM REQUIREMENTS

### Tối thiểu (Testing):
- Python 3.8+
- 8GB RAM
- CPU only

### Khuyến nghị (Training):
- Python 3.10+
- 32GB RAM
- NVIDIA GPU 16GB+ VRAM
- CUDA 11.8+

---

## 📊 CHECKLIST

### Cài đặt:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`install.bat`)
- [ ] `test_import.py` passed

### Validation:
- [ ] `test_training_pipeline.py` → 6/6 passed
- [ ] Ready to train/test

### Training (Optional):
- [ ] Dataset prepared (20-50 images)
- [ ] `run_training.bat` executed
- [ ] Model saved to `output/`

### Testing (Optional):
- [ ] BrushNet checkpoint downloaded
- [ ] `test_hybrid.py` working
- [ ] Output images generated

---

## 🆘 TROUBLESHOOTING

### Installation issues:
→ See `INSTALLATION.md`

### Test failures:
→ See `TESTING_GUIDE.md`

### Training errors:
→ See `PROJECT_COMPLETE.md`

### Code questions:
→ See `CODE_REVIEW.md` & `ARCHITECTURE.md`

---

## 🎓 LEARNING PATH

### Beginner:
1. Install → `INSTALL_QUICK.md`
2. Validate → `QUICK_TEST_GUIDE.md`
3. Understand → `README.md`

### Intermediate:
1. Complete guide → `PROJECT_COMPLETE.md`
2. Architecture → `ARCHITECTURE.md`
3. Train model → `run_training.bat`

### Advanced:
1. Code review → `CODE_REVIEW.md`
2. Customize pipeline → `pipelines/pipeline_hybrid_sd_xl.py`
3. Modify UNet → `models/unets/unet_2d_condition.py`

---

## 🔗 EXTERNAL RESOURCES

- **BrushNet**: https://github.com/TencentARC/BrushNet
- **Diffusers**: https://github.com/huggingface/diffusers
- **SDXL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

---

## 📞 NEXT STEPS

**Sau khi đọc file này:**

1. ✅ Cài đặt: Chạy `install.bat`
2. ✅ Validate: Chạy `quick_validate.bat`
3. ✅ Đọc thêm: `README.md` hoặc `PROJECT_COMPLETE.md`
4. ✅ Train/Test: Tùy mục đích của bạn

---

**Project Status**: ✅ Ready for use  
**Last Updated**: 2026-02-06  
**Version**: 1.0.0

**Enjoy! 🎨**
