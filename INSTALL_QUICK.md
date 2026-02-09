# 🚀 CÁCH CÀI ĐẶT NHANH NHẤT

## Cho người vội (Copy & Paste)

```bash
# 1. Mở terminal/cmd tại thư mục InteriorInpaint
cd E:\final_project\Task-2\InteriorInpaint

# 2. Tạo và activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Chạy install script
install.bat
```

**Xong!** Script sẽ tự động cài đặt tất cả.

---

## Hoặc cài thủ công (3 lệnh):

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Verify cài đặt thành công

```bash
python test_import.py
```

**Thấy "All imports successful!"** → OK!

---

## Full checklist

- [x] Python 3.8+ installed
- [x] Virtual environment created
- [x] PyTorch with CUDA installed
- [x] Requirements installed
- [x] `test_import.py` passed

→ **Chạy validation**: `python test_training_pipeline.py`

---

## Lỗi thường gặp

### "CUDA not available"
→ PyTorch CPU-only. Cài lại:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "ModuleNotFoundError: diffusers"
→ Chưa cài requirements:
```bash
pip install -r requirements.txt
```

---

**Chi tiết**: Xem `INSTALLATION.md`
