
import sys
import os

# Add the project root to sys.path
sys.path.append(r"E:\final_project\Task-2")

try:
    from InteriorInpaint.pipelines import StableDiffusionXLHybridPipeline
    from InteriorInpaint.models import BrushNetModel, UNet2DConditionModel
    print("Successfully imported StableDiffusionXLHybridPipeline and custom models.")
except Exception as e:
    print(f"Failed to import: {e}")
    import traceback
    traceback.print_exc()
