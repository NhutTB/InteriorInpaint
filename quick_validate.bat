@echo off
REM Quick validation script for training pipeline

echo ========================================
echo Training Pipeline Validation
echo ========================================
echo.

echo Running validation tests...
echo.

python test_training_pipeline.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS! Ready to train.
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Prepare your dataset (20-50 images)
    echo   2. Run: run_training.bat
    echo.
) else (
    echo.
    echo ========================================
    echo VALIDATION FAILED
    echo ========================================
    echo.
    echo Please fix the issues above before training.
    echo.
)

pause
