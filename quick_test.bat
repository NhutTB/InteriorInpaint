@echo off
REM Quick test script for InteriorInpaint Hybrid Pipeline
REM Edit paths below before running

echo ========================================
echo InteriorInpaint - Quick Test
echo ========================================
echo.

REM ========== CONFIGURATION ==========

REM Base SDXL model
set BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0

REM BrushNet checkpoint (download from BrushNet repo)
set BRUSHNET_PATH=data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v0

REM ControlNet checkpoint
set CONTROLNET_PATH=diffusers/controlnet-mlsd-sdxl-1.0

REM Input images
set IMAGE_PATH=examples/test_image.jpg
set MASK_PATH=examples/test_mask.jpg

REM Prompt
set PROMPT=modern minimalist sofa with white cushions in a bright living room

REM ========== RUN TEST ==========

echo [1] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2] Testing imports...
python test_import.py
if %errorlevel% neq 0 (
    echo ERROR: Import test failed!
    echo Please install dependencies: pip install diffusers transformers accelerate torch
    pause
    exit /b 1
)

echo.
echo [3] Running hybrid pipeline test...
echo    Base Model: %BASE_MODEL%
echo    BrushNet: %BRUSHNET_PATH%
echo    ControlNet: %CONTROLNET_PATH%
echo    Image: %IMAGE_PATH%
echo    Mask: %MASK_PATH%
echo.

python test_hybrid.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Test failed!
    echo ========================================
    echo.
    echo Possible issues:
    echo  1. Models not downloaded
    echo  2. Insufficient VRAM (need 16GB+)
    echo  3. Missing dependencies
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Check output_hybrid.png
echo ========================================
echo.
pause
