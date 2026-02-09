@echo off
REM A/B Comparison Test for Hybrid Pipeline

echo ========================================
echo A/B Testing: Hybrid vs BrushNet-only
echo ========================================
echo.

echo Creating test data directory...
mkdir test_data 2>nul
mkdir test_output 2>nul
mkdir test_output\ab_comparison 2>nul

echo.
echo Running A/B test...
python test_ab_comparison.py

echo.
echo Done! Check test_output\ab_comparison\ for results.
pause
