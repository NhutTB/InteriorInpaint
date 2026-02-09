@echo off
REM Training script for DreamBooth Interior Fine-tuning
REM Edit paths and parameters below

echo ========================================
echo InteriorInpaint - DreamBooth Training
echo ========================================
echo.

REM ========== CONFIGURATION ==========

REM Base model
set PRETRAINED_MODEL=stabilityai/stable-diffusion-xl-base-1.0

REM VAE (fp16 fix - IMPORTANT!)
set VAE_MODEL=madebyollin/sdxl-vae-fp16-fix

REM Dataset path (folder with your interior images)
set DATA_DIR=data/interior_images

REM Instance prompt (describe your style)
set INSTANCE_PROMPT=a photo of modern minimalist interior design

REM Output directory
set OUTPUT_DIR=output/dreambooth_interior

REM Training parameters
set RESOLUTION=1024
set BATCH_SIZE=1
set GRAD_ACCUMULATION=4
set LEARNING_RATE=1e-6
set NUM_EPOCHS=100
set CHECKPOINT_STEPS=500

REM ========== START TRAINING ==========

echo Configuration:
echo   Model: %PRETRAINED_MODEL%
echo   VAE: %VAE_MODEL%
echo   Dataset: %DATA_DIR%
echo   Prompt: %INSTANCE_PROMPT%
echo   Output: %OUTPUT_DIR%
echo   Epochs: %NUM_EPOCHS%
echo   Learning Rate: %LEARNING_RATE%
echo.

echo Checking dataset...
if not exist "%DATA_DIR%" (
    echo ERROR: Dataset directory not found: %DATA_DIR%
    echo Please create this folder and add your training images!
    pause
    exit /b 1
)

echo.
echo Starting training with Accelerate...
echo.

accelerate launch train_dreambooth.py ^
  --pretrained_model_name_or_path="%PRETRAINED_MODEL%" ^
  --pretrained_vae_model_name_or_path="%VAE_MODEL%" ^
  --instance_data_dir="%DATA_DIR%" ^
  --instance_prompt="%INSTANCE_PROMPT%" ^
  --output_dir="%OUTPUT_DIR%" ^
  --resolution=%RESOLUTION% ^
  --train_batch_size=%BATCH_SIZE% ^
  --gradient_accumulation_steps=%GRAD_ACCUMULATION% ^
  --learning_rate=%LEARNING_RATE% ^
  --num_train_epochs=%NUM_EPOCHS% ^
  --checkpointing_steps=%CHECKPOINT_STEPS% ^
  --mixed_precision=fp16 ^
  --gradient_checkpointing

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Training failed!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed successfully!
echo Model saved to: %OUTPUT_DIR%
echo ========================================
echo.
pause
