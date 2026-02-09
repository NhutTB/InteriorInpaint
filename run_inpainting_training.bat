@echo off
REM Supervised Inpainting Training Script
REM Edit paths below before running

echo ========================================
echo Supervised Inpainting Training
echo ========================================
echo.

REM ====== CONFIGURATION ======
set SDXL_MODEL=stabilityai/stable-diffusion-xl-base-1.0
set VAE_MODEL=madebyollin/sdxl-vae-fp16-fix
set DATA_DIR=data\inpainting_pairs
set OUTPUT_DIR=output\inpainting_model

REM Optional: pretrained BrushNet
set BRUSHNET_PATH=

REM Training: What to train
set TRAIN_UNET=
set TRAIN_BRUSHNET=--train_brushnet

REM Training params
set EPOCHS=100
set BATCH_SIZE=1
set LEARNING_RATE=1e-5
set RESOLUTION=1024

REM ====== END CONFIGURATION ======

echo Configuration:
echo   SDXL Model: %SDXL_MODEL%
echo   Data Dir: %DATA_DIR%
echo   Output: %OUTPUT_DIR%
echo   Train BrushNet: %TRAIN_BRUSHNET%
echo   Train UNet: %TRAIN_UNET%
echo.

REM Check data directory
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR%
    echo.
    echo Please create dataset with structure:
    echo   %DATA_DIR%\
    echo   ├── inputs\room1_masked.jpg
    echo   ├── targets\room1.jpg
    echo   └── metadata.jsonl
    echo.
    pause
    exit /b 1
)

echo Starting training...
echo.

accelerate launch train_inpainting.py ^
    --pretrained_model_name_or_path="%SDXL_MODEL%" ^
    --pretrained_vae_model_name_or_path="%VAE_MODEL%" ^
    --data_dir="%DATA_DIR%" ^
    --output_dir="%OUTPUT_DIR%" ^
    --resolution=%RESOLUTION% ^
    --train_batch_size=%BATCH_SIZE% ^
    --num_train_epochs=%EPOCHS% ^
    --learning_rate=%LEARNING_RATE% ^
    --gradient_accumulation_steps=4 ^
    --gradient_checkpointing ^
    --mixed_precision=fp16 ^
    --checkpointing_steps=500 ^
    --logging_steps=10 ^
    %TRAIN_BRUSHNET% %TRAIN_UNET%

echo.
echo ========================================
echo Training Complete!
echo Model saved to: %OUTPUT_DIR%
echo ========================================
pause
