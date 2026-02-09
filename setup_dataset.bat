@echo off
REM Quick dataset setup script

echo ========================================
echo Dataset Setup Helper
echo ========================================
echo.

set /p DATASET_NAME="Enter dataset name (e.g., modern_interior): "
set DATASET_PATH=data\%DATASET_NAME%

echo Creating dataset folder: %DATASET_PATH%
mkdir "%DATASET_PATH%" 2>nul

if exist "%DATASET_PATH%" (
    echo.
    echo ✅ Dataset folder created successfully!
    echo.
    echo Next steps:
    echo   1. Copy your images to: %DATASET_PATH%
    echo   2. Supported formats: .jpg, .png, .jpeg, .webp
    echo   3. Recommended: 20-50 images
    echo   4. Resolution: Any (will be resized to 1024x1024)
    echo.
    echo Example structure:
    echo   %DATASET_PATH%\
    echo   ├── image1.jpg
    echo   ├── image2.png
    echo   └── ...
    echo.
    echo After adding images, run:
    echo   run_training.bat
    echo.
    echo And edit these lines in run_training.bat:
    echo   set DATA_DIR=%DATASET_PATH%
    echo   set INSTANCE_PROMPT=a photo of your style here
    echo.
    
    explorer "%DATASET_PATH%"
) else (
    echo ❌ Failed to create folder
)

pause
