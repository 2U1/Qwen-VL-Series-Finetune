"""
Shared vLLM client module for model inference.

Extracted from gui_perturbed_evaluator.py for reuse across evaluation scripts.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from PIL import Image
from loguru import logger

# Import prompt builders
from prompts import (
    build_gta1_messages,
    build_uitars15_messages,
    build_qwen25vl_messages,
)

# Import constants
from prompts import EXPECTED_IMAGE_WIDTH, EXPECTED_IMAGE_HEIGHT

# Constants for image processing
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model inference configuration."""
    name: str  # Model identifier
    model_type: str  # gta1, qwen25vl, uitars15
    use_reasoning: bool  # Whether to use reasoning prompt template
    temperature: float = 0.0
    max_tokens: int = 1000
    top_p: float = 0.9
    seed: Optional[int] = None
    language: str = "English"
    image_factor: int = 28  # Image resize factor (patch_size * merge_size)
    image_min_pixels: int = MIN_PIXELS
    image_max_pixels: int = MAX_PIXELS  # Maximum image pixels


# ============================================================================
# Helper Functions
# ============================================================================

def format_metadata_string(task_id: Optional[str] = None, 
                           step_index: Optional[int] = None, 
                           variant: Optional[str] = None) -> str:
    """Format metadata for logging."""
    if task_id is None and step_index is None and variant is None:
        return ""
    parts = []
    if task_id is not None:
        parts.append(f"task_id={task_id}")
    if step_index is not None:
        parts.append(f"step_index={step_index}")
    if variant is not None:
        parts.append(f"variant={variant}")
    return f" [{', '.join(parts)}]"


# ============================================================================
# Model Client
# ============================================================================

class ModelClient:
    """Client for vLLM API inference."""
    
    def __init__(self, config: ModelConfig, api_url: str, api_key: str):
        self.config = config
        self.client = OpenAI(base_url=api_url, api_key=api_key)
    
    def predict(self, instruction: str, image_path: Path, 
               metadata: Optional[Dict[str, Any]] = None,
               use_pattern_matching: bool = False) -> str:
        """
        Run model inference on instruction and image.
        
        Args:
            instruction: Text instruction
            image_path: Path to image file
            metadata: Optional dict with task_id, step_index, variant for logging
            use_pattern_matching: If True, use pattern matching for image loading (for CSV-based evaluation).
                                  If False, load image directly from path (for ScreenSpot).
        
        Returns raw prediction text from model.
        """
        # Load and process image
        metadata = metadata or {}
        image = self._load_image(image_path, use_pattern_matching=use_pattern_matching, **metadata)
        
        # Build messages
        messages = self.build_messages(instruction, image, self.config.model_type, self.config.use_reasoning)
        
        # Make API request
        request_kwargs = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed
        
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content.strip()

    def build_messages(self, instruction: str, image: Image.Image, model_type: str, use_reasoning: bool) -> List[Dict[str, Any]]:
        """Build messages for model inference.""" 
        if model_type == "gta1":
            return build_gta1_messages(instruction, image, use_reasoning)
        elif model_type == "uitars15":
            return build_uitars15_messages(instruction, image, use_reasoning)
        elif model_type == "qwen25vl":
            return build_qwen25vl_messages(instruction, image, use_reasoning)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    
    def _load_image(self, image_path: Path, 
                    use_pattern_matching: bool = False,
                    task_id: Optional[str] = None, 
                    step_index: Optional[int] = None, 
                    variant: Optional[str] = None) -> Image.Image:
        """
        Load, validate, and resize image using smart_resize.
        
        Args:
            image_path: Path to image file
            use_pattern_matching: If True, use pattern matching to find image file (for CSV-based evaluation).
                                  If False, load image directly from path (for ScreenSpot).
            task_id: Optional task ID for logging
            step_index: Optional step index for logging (required if use_pattern_matching=True)
            variant: Optional variant for logging
        
        Returns:
            PIL Image object (not resized, original dimensions preserved for coordinate normalization)
        """
        if use_pattern_matching:
            # Pattern matching logic for CSV-based evaluation (gui_perturbed_evaluator)
            # the image_path can be inaccurate with the final file name which has the format of step_<index>_<action>.png
            # and the action can be wrong, so we need to get the correct image path from the task_id and step_index
            image_folder = image_path.parent
            # use step index and the image folder only because image filename in the csv file sometimes has the wrong action name in the filename.
            search_pattern = f"step_{step_index}_*.png"
            image_files = list(image_folder.glob(search_pattern))
            
            if len(image_files) == 0:
                raise FileNotFoundError(
                    f"Image files not found: pattern '{search_pattern}' in folder {image_folder} "
                    f"for task {task_id} and step {step_index}"
                )
            
            image_file = image_files[0]
            if len(image_files) > 1:
                logger.warning(f"Multiple images found for task {task_id} step {step_index}, using: {image_file}")
        else:
            # Direct image loading for ScreenSpot
            image_file = image_path
        
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Store original dimensions
        original_width, original_height = image.size
        
        # Check if image is 1920x1080 (expected resolution) - only warn, don't fail
        if original_width != EXPECTED_IMAGE_WIDTH or original_height != EXPECTED_IMAGE_HEIGHT:
            metadata_str = format_metadata_string(task_id, step_index, variant)
            logger.warning(
                f"[Image Dimension Check] Image is not {EXPECTED_IMAGE_WIDTH}x{EXPECTED_IMAGE_HEIGHT}: "
                f"actual={original_width}x{original_height}{metadata_str} path={image_file}"
            )

        return image

