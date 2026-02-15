@echo off
REM Training script OPTIMIZED FOR 8GB VRAM (local testing)
REM Use this for testing code locally before training on server

echo ========================================
echo Sequential Image Editing Training (8GB VRAM)
echo ========================================
echo.

REM ====== CONFIGURATION ======
set HF_TOKEN="hf_xxxxxx"
set SDXL_MODEL=stabilityai/stable-diffusion-xl-base-1.0
set VAE_MODEL=madebyollin/sdxl-vae-fp16-fix
set DATA_DIR=E:\final_project\Task-2\InteriorInpaint\test
set OUTPUT_DIR=output\sequential_editing_test_8gb

REM Train ONE module only for 8GB
set TRAIN_UNET=
set TRAIN_BRUSHNET=--train_brushnet

REM 8GB VRAM optimizations
set BATCH_SIZE=1
set GRAD_ACCUM=4
set RESOLUTION=512
set MAX_STEPS=10
set LEARNING_RATE=1e-5

REM Checkpointing
set CHECKPOINT_STEPS=500
set CHECKPOINTS_LIMIT=3
set RESUME=

REM Validation (less frequent to save VRAM)
set VALIDATION_STEPS=1000
set NUM_VALIDATION=2

REM Wandb
set USE_WANDB=--use_wandb
set WANDB_PROJECT=sequential-image-editing

REM Dataset
set USE_GLOBAL_CAPTIONS=--use_global_captions

echo Configuration (8GB optimized):
echo   Resolution: %RESOLUTION%
echo   Batch Size: %BATCH_SIZE%
echo   Gradient Accumulation: %GRAD_ACCUM%
echo   Train: BrushNet only
echo.

REM Check data
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR%
    pause
    exit /b 1
)

echo Starting training...
echo.

accelerate launch train_sequential_editing.py ^
    --pretrained_model_name_or_path="%SDXL_MODEL%" ^
    --pretrained_vae_model_name_or_path="%VAE_MODEL%" ^
    --data_dir="%DATA_DIR%" ^
    --output_dir="%OUTPUT_DIR%" ^
    --resolution=%RESOLUTION% ^
    --train_batch_size=%BATCH_SIZE% ^
    --max_train_steps=%MAX_STEPS% ^
    --learning_rate=%LEARNING_RATE% ^
    --gradient_accumulation_steps=%GRAD_ACCUM% ^
    --gradient_checkpointing ^
    --mixed_precision=fp16 ^
    --checkpointing_steps=%CHECKPOINT_STEPS% ^
    --checkpoints_total_limit=%CHECKPOINTS_LIMIT% ^
    --validation_steps=%VALIDATION_STEPS% ^
    --num_validation_images=%NUM_VALIDATION% ^
    --logging_steps=10 ^
    %TRAIN_BRUSHNET% ^
    %USE_GLOBAL_CAPTIONS% ^
    %RESUME% ^
    %USE_WANDB% --wandb_project=%WANDB_PROJECT%

echo.
echo Training Complete!
pause
