# 🎯 DATASET & PRETRAINED - TL;DR

## 📁 Cấu trúc Dataset (CỰC KỲ ĐơnN GIẢN!)

```
data/
└── interior_images/          ← Chỉ cần 1 folder!
    ├── image1.jpg           ← 20-50 ảnh
    ├── image2.png           ← Bất kỳ format nào
    ├── image3.jpg
    └── ...
```

**Chỉ cần copy ảnh vào 1 folder. XONG!**

---

## ✅ Dùng Pretrained Weights? 

**CÓ! Script mặc định ĐÃ DÙNG pretrained SDXL!**

```bash
# Script tự động load:
stabilityai/stable-diffusion-xl-base-1.0
```

→ **KHÔNG CẦN làm gì thêm!**

---

## 🚀 Quick Start (3 bước)

### 1. Tạo dataset folder
```bash
setup_dataset.bat
# Nhập tên → Folder tự tạo → Copy ảnh vào
```

### 2. Sửa config (Optional)
```bash
# Mở run_training.bat, sửa:
set DATA_DIR=data/interior_images
set INSTANCE_PROMPT=a photo of modern interior
```

### 3. Train!
```bash
run_training.bat
```

**XONG!**

---

## 💡 FAQs

### Q: Train từ đâu?
**A**: Mặc định từ **SDXL pretrained** (tự động download)

### Q: Cần bao nhiêu ảnh?
**A**: **20-50 ảnh** là đủ (nhờ pretrained)

### Q: Ảnh phải như nào?
**A**: 
- ✅ Resolution >= 512px
- ✅ Rõ nét
- ✅ Style nhất quán
- ✅ Format: jpg/png/webp

### Q: Train mất bao lâu?
**A**: 
- 30 ảnh, 100 epochs: ~2-4 giờ (RTX 3090)
- 50 ảnh, 100 epochs: ~4-6 giờ

### Q: Resume training được không?
**A**: Được!
```bash
# Trong run_training.bat, thêm:
--resume_from_checkpoint="output/my_model/checkpoint-500"
```

### Q: Dùng model khác SDXL được không?
**A**: Được! Sửa trong `run_training.bat`:
```bash
set PRETRAINED_MODEL=SG161222/RealVisXL_V3.0
```

---

## 📖 Đọc thêm

- Chi tiết: `DATASET_GUIDE.md`
- Training options: `train_dreambooth.py --help`
- Full guide: `PROJECT_COMPLETE.md`

---

**Bottom line**: 
1. Copy ảnh vào `data/interior_images/`
2. Chạy `run_training.bat`
3. Đợi 2-4 giờ
4. XONG! ✅
