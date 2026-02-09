@echo off
REM Quick installation script for InteriorInpaint
REM Run this after activating your virtual environment

echo ========================================
echo InteriorInpaint - Quick Installation
echo ========================================
echo.

echo [Step 1/5] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo [Step 2/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [Step 3/5] Installing PyTorch (CUDA 11.8)...
echo This may take a few minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo WARNING: PyTorch installation failed. Trying CPU version...
    pip install torch torchvision
)
echo.

echo [Step 4/5] Installing core requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Requirements installation failed!
    pause
    exit /b 1
)
echo.

echo [Step 5/5] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
python test_import.py
echo.

if %errorlevel% equ 0 (
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
    echo.
    echo Optional: Install controlnet-aux for preprocessing
    echo   pip install controlnet-aux
    echo.
    echo Optional: Install xformers for faster training
    echo   pip install xformers
    echo.
    echo Next step: Run validation test
    echo   python test_training_pipeline.py
    echo.
) else (
    echo ========================================
    echo Installation completed with warnings
    echo ========================================
    echo Please check errors above
    echo.
)

pause
