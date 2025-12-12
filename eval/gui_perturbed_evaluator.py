"""
Standalone CSV-based evaluation script.

Loads evaluation data from CSV, runs model inference, and saves raw predictions.
"""

import argparse
import json
import os
import sys
import math
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from io import BytesIO

import pandas as pd
from openai import OpenAI
from PIL import Image
from loguru import logger

# Add eval directory to path for imports
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from prompts import (
    UITARS_ACTION_SPACE,
    UITARS_USR_PROMPT_THOUGHT,
    UITARS_USR_PROMPT_NOTHOUGHT,
    GTA1_SYSTEM_PROMPT,
    render_qwen25_tools_system,
)


# ============================================================================
# Constants
# ============================================================================

EXPECTED_IMAGE_WIDTH = 1920
EXPECTED_IMAGE_HEIGHT = 1080

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VALID_MODEL_TYPES = {"gta1", "qwen25vl", "uitars15"}

try:
    from qwen_vl_utils import smart_resize
except ImportError:
    # Fallback implementation when qwen_vl_utils is not available
    def _round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def _ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def _floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer <= 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
        """Rescale image dimensions to meet constraints."""
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(f"Aspect ratio must be < {MAX_RATIO}")
        h_bar = max(factor, _round_by_factor(height, factor))
        w_bar = max(factor, _round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = _floor_by_factor(height / beta, factor)
            w_bar = _floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = _ceil_by_factor(height * beta, factor)
            w_bar = _ceil_by_factor(width * beta, factor)
        return h_bar, w_bar


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

class DatasetVariantType(Enum):
    STYLE = "style"
    PRECISION = "precision"
    TEXT_ZOOM = "text_zoom"
    ORIGINAL = "original"

class InstructionType(Enum):
    DIRECT_QUERY = "direct_query"
    RELATIONAL_QUERY = "relational_query"

@dataclass
class DatasetConfig:
    """Dataset variant configuration."""
    dataset_variant: DatasetVariantType
    instruction_type: InstructionType

@dataclass
class EvaluationConfig:
    """Overall evaluation configuration."""
    csv_path: Path
    screenshots_base_dir: Path
    output_dir: Path
    model_config: ModelConfig
    dataset_config: DatasetConfig
    api_url: str
    api_key: str
    save_interval: int = 10  # Save predictions every N steps


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


def setup_logging(output_dir: Path) -> Path:
    """Set up logging to both console and file. Returns log file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_{timestamp}.log"
    log_path = output_dir / log_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (INFO level)
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # Add file handler (DEBUG level)
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    
    return log_path


# ============================================================================
# Data Loader
# ============================================================================

class DataLoader:
    """Loads and filters evaluation data from CSV."""
    
    def __init__(self, csv_path: Path, dataset_config: DatasetConfig, screenshots_base_dir: Path):
        self.csv_path = csv_path
        self.dataset_config = dataset_config
        self.screenshots_base_dir = screenshots_base_dir
        self.df = self._load_and_filter()
    
    def _load_and_filter(self) -> pd.DataFrame:
        """Load CSV and filter by dataset variant configuration."""
        df = pd.read_csv(self.csv_path)
        
        # Filter by dataset variant type
        variant_value = self.dataset_config.dataset_variant.value
        df = df[df["variant"] == variant_value]
        
        # Filter by instruction type
        instruction_col = (
            "step_instruction" 
            if self.dataset_config.instruction_type == InstructionType.DIRECT_QUERY 
            else "multi_element_instruction"
        )
        df = df[df[instruction_col].notna() & (df[instruction_col] != "")]
        
        return df.sort_values(["task_id", "step_index"]).reset_index(drop=True)
    
    def get_rows(self) -> List[Dict]:
        """Get all filtered rows with resolved screenshot paths."""
        rows = self.df.to_dict("records")
        
        for row in rows:
            # Get instruction based on instruction type
            if self.dataset_config.instruction_type == InstructionType.DIRECT_QUERY:
                row["instruction"] = row.get("step_instruction", "")
            else:
                row["instruction"] = row.get("multi_element_instruction", "")
        
        return rows


# ============================================================================
# Model Client
# ============================================================================

class ModelClient:
    """Client for vLLM API inference."""
    
    def __init__(self, config: ModelConfig, api_url: str, api_key: str):
        self.config = config
        self.client = OpenAI(base_url=api_url, api_key=api_key)
    
    def predict(self, instruction: str, image_path: Path, 
               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Run model inference on instruction and image.
        
        Args:
            instruction: Text instruction
            image_path: Path to image file
            metadata: Optional dict with task_id, step_index, variant for logging
        
        Returns raw prediction text from model.
        """
        # Load and process image
        metadata = metadata or {}
        image = self._load_image(image_path, **metadata)
        image_base64 = self._encode_image(image)
        
        # Get prompts
        system_text, user_text = self._get_prompts(instruction, image)
        
        # Build messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ]
        
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
    
    def _load_image(self, image_path: Path, 
                    task_id: Optional[str] = None, 
                    step_index: Optional[int] = None, 
                    variant: Optional[str] = None) -> Image.Image:
        """
        Load, validate, and resize image using smart_resize.
        
        Args:
            image_path: Path to image file
            task_id: Optional task ID for logging
            step_index: Optional step index for logging
            variant: Optional variant for logging
        
        Returns:
            Resized image ready for inference
        """
        with image_path.open("rb") as f:
            image_bytes = f.read()
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Store original dimensions
        original_width, original_height = image.size
        
        # Check if image is 1920x1080 (expected resolution)
        if original_width != EXPECTED_IMAGE_WIDTH or original_height != EXPECTED_IMAGE_HEIGHT:
            metadata_str = format_metadata_string(task_id, step_index, variant)
            logger.warning(
                f"[Image Dimension Check] Image is not {EXPECTED_IMAGE_WIDTH}x{EXPECTED_IMAGE_HEIGHT}: "
                f"actual={original_width}x{original_height}{metadata_str} path={image_path}"
            )
        
        # Apply smart resize
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=self.config.image_factor,
            min_pixels=self.config.image_min_pixels,
            max_pixels=self.config.image_max_pixels,
        )
        
        # Resize the image
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        
        logger.debug(f"[Image] Original size: {original_width}x{original_height}")
        logger.debug(f"[Image] Resized size: {resized_width}x{resized_height}")
        
        return resized_image
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _get_prompts(self, instruction: str, image: Image.Image) -> Tuple[str, str]:
        """
        Get system and user prompts based on model config.
        
        Prompt selection logic:
        - gta1: Uses GTA1_SYSTEM_PROMPT (use_reasoning is ignored)
        - qwen25vl: Uses UITARS prompts (controlled by use_reasoning)
        - uitars15: Uses UITARS prompts (controlled by use_reasoning)
        
        Returns:
            (system_prompt, user_prompt)
        """
        model_type = self.config.model_type.lower().strip()
        img_w, img_h = image.size
        
        # Log model_type for debugging
        logger.debug(f"[Prompt] Config model_type: '{self.config.model_type}', normalized: '{model_type}'")
        
        # GTA1 model - use resized dimensions for system prompt
        if model_type == "gta1":
            system_text = GTA1_SYSTEM_PROMPT.format(height=img_h, width=img_w)
            user_text = instruction
            self._log_prompt_selection("GTA1_SYSTEM_PROMPT", model_type, system_text, user_text, 
                                     use_reasoning_ignored=True)
            return system_text, user_text
        
        # UITARS prompts for qwen25vl and uitars15
        if model_type in ("qwen25vl", "uitars15"):
            if self.config.use_reasoning:
                template_name = "UITARS_USR_PROMPT_THOUGHT"
                user_text = UITARS_USR_PROMPT_THOUGHT.format(
                    action_space=UITARS_ACTION_SPACE,
                    language=self.config.language,
                    instruction=instruction,
                )
            else:
                template_name = "UITARS_USR_PROMPT_NOTHOUGHT"
                user_text = UITARS_USR_PROMPT_NOTHOUGHT.format(
                    action_space=UITARS_ACTION_SPACE,
                    instruction=instruction,
                )
            system_text = "You are a helpful assistant."
            self._log_prompt_selection(template_name, model_type, system_text, user_text)
            return system_text, user_text
        
        # Default fallback - should not happen with valid presets
        logger.error(
            f"[Prompt] Unknown model type: '{model_type}' (from config.model_type='{self.config.model_type}'). "
            f"Using default fallback. Valid types: {VALID_MODEL_TYPES}"
        )
        system_text = "You are a helpful assistant."
        user_text = instruction
        self._log_prompt_selection("DEFAULT", model_type, system_text, user_text)
        return system_text, user_text
    
    def _log_prompt_selection(self, template_name: str, model_type: str, 
                              system_text: str, user_text: str, 
                              use_reasoning_ignored: bool = False):
        """Log prompt selection details."""
        reasoning_info = "(ignored for GTA1)" if use_reasoning_ignored else f"use_reasoning={self.config.use_reasoning}"
        logger.debug(f"[Prompt] Using template: {template_name}")
        logger.debug(f"[Prompt] Model type: {model_type}, {reasoning_info}")
        logger.debug(f"[Prompt] System prompt: {system_text[:200]}...")
        logger.debug(f"[Prompt] User prompt (first 200 chars): {user_text[:200]}...")


# ============================================================================
# Prediction Saver
# ============================================================================

class PredictionSaver:
    """Saves predictions to JSONL file with frequent flushing."""
    
    def __init__(self, output_path: Path, save_interval: int = 10):
        self.output_path = output_path
        self.save_interval = save_interval
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = output_path.open("w", encoding="utf-8")
        self.count = 0
    
    def save(self, prediction: Dict):
        """Save a single prediction and flush if needed."""
        self.file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        self.count += 1
        
        if self.count % self.save_interval == 0:
            self.file.flush()
    
    def close(self):
        """Close the output file."""
        self.file.close()


# ============================================================================
# Evaluation Runner
# ============================================================================

class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.data_loader = DataLoader(
            config.csv_path,
            config.dataset_config,
            config.screenshots_base_dir
        )
        self.model_client = ModelClient(
            config.model_config,
            config.api_url,
            config.api_key
        )
        self.saver = PredictionSaver(
            self._get_output_path(),
            config.save_interval
        )
    
    def _get_output_path(self) -> Path:
        """Generate output file path based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"predictions_"
            f"{self.config.model_config.model_type}_"
            f"{'reasoning' if self.config.model_config.use_reasoning else 'no_reasoning'}_"
            f"{self.config.dataset_config.dataset_variant.value}_"
            f"{self.config.dataset_config.instruction_type.value}_"
            f"{timestamp}.jsonl"
        )
        return self.config.output_dir / filename
    
    def run(self):
        """Run evaluation on all CSV rows."""
        rows = self.data_loader.get_rows()
        total_rows = len(rows)
        
        logger.info(f"Starting evaluation on {total_rows} rows")
        
        for idx, row in enumerate(rows, 1):
            prediction = self._process_row(row)
            self.saver.save(prediction)
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{total_rows} rows ({idx/total_rows*100:.1f}%)")
        
        self.saver.close()
        logger.info(f"Evaluation completed. Processed {total_rows} rows")
    
    def _process_row(self, row: Dict) -> Dict:
        """Process a single CSV row and return prediction."""
        instruction = row["instruction"]
        image_path = self.data_loader.screenshots_base_dir / row["image_path"]
        
        # Prepare metadata for logging
        metadata = {
            "task_id": row.get("task_id"),
            "step_index": row.get("step_index"),
            "variant": row.get("variant"),
        }
        
        raw_prediction = self.model_client.predict(instruction, image_path, metadata=metadata)
        
        return {
            "task_id": row["task_id"],
            "step_index": row["step_index"],
            "instruction": instruction,
            "image_path": str(image_path),
            "raw_prediction": raw_prediction,
            "model_config": {
                "name": self.config.model_config.name,
                "model_type": self.config.model_config.model_type,
                "use_reasoning": self.config.model_config.use_reasoning,
            },
            "dataset_config": {
                "dataset_variant": self.config.dataset_config.dataset_variant.value,
                "instruction_type": self.config.dataset_config.instruction_type.value,
            },
        }


# ============================================================================
# Predefined Configurations
# ============================================================================

@dataclass
class EvaluationPreset:
    """Predefined evaluation configuration preset."""
    config_id: str
    model_type: str
    use_reasoning: bool
    dataset_variant: DatasetVariantType
    instruction_type: InstructionType


def _create_preset(
    model_type: str,
    use_reasoning: bool,
    dataset_variant: DatasetVariantType,
    instruction_type: InstructionType,
) -> EvaluationPreset:
    """Create a single preset with explicit config_id."""
    reasoning_str = "reasoning" if use_reasoning else "no_reasoning"
    config_id = f"{model_type}_{reasoning_str}_{dataset_variant.value}_{instruction_type.value}"
    
    return EvaluationPreset(
        config_id=config_id,
        model_type=model_type,
        use_reasoning=use_reasoning,
        dataset_variant=dataset_variant,
        instruction_type=instruction_type,
    )


def _generate_all_presets() -> Dict[str, EvaluationPreset]:
    """Generate all possible evaluation configuration presets.
    
    Generates 48 total combinations:
    - 3 models (gta1, qwen25vl, uitars15)
    - 2 reasoning modes (with/without)
    - 4 dataset variants (style, precision, text_zoom, original)
    - 2 instruction types (direct_query, relational_query)
    
    This explicit structure makes it easy to verify correctness and debug.
    """
    presets = {}
    
    # Define all model types explicitly
    MODEL_TYPES = ["gta1", "qwen25vl", "uitars15"]
    
    # Define all other dimensions explicitly
    REASONING_MODES = [False, True]
    DATASET_VARIANTS = [
        DatasetVariantType.STYLE,
        DatasetVariantType.PRECISION,
        DatasetVariantType.TEXT_ZOOM,
        DatasetVariantType.ORIGINAL,
    ]
    INSTRUCTION_TYPES = [
        InstructionType.DIRECT_QUERY,
        InstructionType.RELATIONAL_QUERY,
    ]
    
    # Generate all combinations
    for model_type in MODEL_TYPES:
        for use_reasoning in REASONING_MODES:
            for dataset_variant in DATASET_VARIANTS:
                for instruction_type in INSTRUCTION_TYPES:
                    preset = _create_preset(
                        model_type=model_type,
                        use_reasoning=use_reasoning,
                        dataset_variant=dataset_variant,
                        instruction_type=instruction_type,
                    )
                    presets[preset.config_id] = preset
    
    return presets


EVALUATION_PRESETS = _generate_all_presets()


def list_presets() -> List[str]:
    """List all available preset configuration IDs."""
    return sorted(EVALUATION_PRESETS.keys())


def get_preset(config_id: str) -> EvaluationPreset:
    """Get a preset configuration by ID."""
    if config_id not in EVALUATION_PRESETS:
        available = ", ".join(list_presets()[:10])
        raise ValueError(
            f"Unknown config_id: {config_id}. "
            f"Available presets (showing first 10): {available}... "
            f"Use --list_presets to see all."
        )
    return EVALUATION_PRESETS[config_id]


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="evaluation script")
    
    # CSV and output
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to CSV file")
    parser.add_argument("--screenshots_base_dir", type=Path, required=True, help="Base directory containing screenshot folders")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    
    # Configuration selection
    parser.add_argument("--config_id", type=str, default=None, help="Preset configuration ID (e.g., 'gta1_no_reasoning_style_direct_query')")
    parser.add_argument("--list_presets", action="store_true", help="List all available preset configuration IDs and exit")
    
    # Model configuration (optional overrides)
    parser.add_argument("--model_name", type=str, default='ByteDance-Seed/UI-TARS-1.5-7B', help="HuggingFace model identifier for vLLM (e.g., 'ByteDance-Seed/UI-TARS-1.5-7B')")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000000000)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--language", type=str, default="English")
    
    # API configuration
    parser.add_argument("--api_url", type=str, default=None, help="API URL (or use VLLM_API_URL env)")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or use VLLM_API_KEY env)")
    
    # Other
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N predictions")
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvaluationConfig:
    """Build evaluation configuration from arguments."""
    if args.list_presets:
        print("Available preset configurations:")
        for preset_id in list_presets():
            preset = EVALUATION_PRESETS[preset_id]
            print(f"  {preset_id}")
            print(f"    Model: {preset.model_type}, Reasoning: {preset.use_reasoning}, "
                  f"Variant: {preset.dataset_variant.value}, Instruction: {preset.instruction_type.value}")
        exit(0)
    
    if args.config_id is None:
        raise ValueError("--config_id is required. Use --list_presets to see available options.")
    
    preset = get_preset(args.config_id)
    
    if args.model_name is None:
        raise ValueError("--model_name is required. Provide the HuggingFace model identifier used by vLLM (e.g., 'ByteDance-Seed/UI-TARS-1.5-7B'). Default is 'ByteDance-Seed/UI-TARS-1.5-7B'.")
    
    # Validate preset model_type
    if preset.model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type '{preset.model_type}' in preset '{args.config_id}'. "
            f"Valid types: {VALID_MODEL_TYPES}"
        )
    
    # Log preset details for debugging
    logger.info(f"Using preset: {args.config_id}")
    logger.info(f"  model_type: {preset.model_type}")
    logger.info(f"  use_reasoning: {preset.use_reasoning}")
    logger.info(f"  dataset_variant: {preset.dataset_variant.value}")
    logger.info(f"  instruction_type: {preset.instruction_type.value}")
    
    model_config = ModelConfig(
        name=args.model_name,
        model_type=preset.model_type,
        use_reasoning=preset.use_reasoning,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
        language=args.language,
    )
    
    # Verify model_type was set correctly
    if model_config.model_type != preset.model_type:
        raise ValueError(
            f"ModelConfig model_type mismatch: expected '{preset.model_type}', "
            f"got '{model_config.model_type}'"
        )
    
    dataset_config = DatasetConfig(
        dataset_variant=preset.dataset_variant,
        instruction_type=preset.instruction_type,
    )
    
    api_url = args.api_url or os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
    api_key = args.api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
    
    return EvaluationConfig(
        csv_path=args.csv_path,
        screenshots_base_dir=args.screenshots_base_dir,
        output_dir=args.output_dir,
        model_config=model_config,
        dataset_config=dataset_config,
        api_url=api_url,
        api_key=api_key,
        save_interval=args.save_interval,
    )


def main():
    """Main entry point."""
    args = parse_args()
    config = build_config(args)
    
    # Set up logging
    log_path = setup_logging(config.output_dir)
    logger.info(f"Logging to file: {log_path}")
    logger.info(f"Starting evaluation with config_id: {args.config_id}")
    
    evaluator = Evaluator(config)
    evaluator.run()
    
    logger.info(f"Evaluation completed. Logs saved to: {log_path}")


"""
uv run eval/gui_perturbed_evaluator.py \
    --csv_path /Users/lockewang/FIG/WebDomainRandomizer/data/variant_data_cleaned.csv \
    --screenshots_base_dir /Users/lockewang/FIG/WebDomainRandomizer/test_splits/ \
    --output_dir data/gui_perturbed_eval/predictions \
    --seed 42 \
    --config_id gta1_no_reasoning_style_direct_query
"""


if __name__ == "__main__":
    main()
