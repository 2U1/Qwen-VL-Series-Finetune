"""
Standalone CSV-based evaluation script.

Loads evaluation data from CSV, runs model inference, and saves raw predictions.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
from openai import OpenAI
from PIL import Image
from loguru import logger

# Add eval directory to path for imports
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from prompts import (
    build_gta1_messages,
    build_uitars15_messages,
    build_qwen25vl_messages,
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

# Model-specific default max_tokens
# GTA1 only needs ~32 tokens for coordinate output (x,y), but we use 64 to be safe
DEFAULT_MAX_TOKENS = {
    ("gta1", False): 64,
    ("gta1", True): 1000,
    ("qwen25vl", False): 1000,
    ("qwen25vl", True): 1000,
    ("uitars15", False): 1000,
    ("uitars15", True): 1000,
}

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
    instruction_type: InstructionType
    dataset_variant: Optional[DatasetVariantType] = None

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
        if self.dataset_config.dataset_variant is not None:
            variant_value = self.dataset_config.dataset_variant.value
            df = df[df["variant"] == variant_value]

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
        
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Store original dimensions
        original_width, original_height = image.size
        
        # Check if image is 1920x1080 (expected resolution)
        if original_width != EXPECTED_IMAGE_WIDTH or original_height != EXPECTED_IMAGE_HEIGHT:
            metadata_str = format_metadata_string(task_id, step_index, variant)
            logger.warning(
                f"[Image Dimension Check] Image is not {EXPECTED_IMAGE_WIDTH}x{EXPECTED_IMAGE_HEIGHT}: "
                f"actual={original_width}x{original_height}{metadata_str} path={image_file}"
            )

        return image



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
        self.output_path = self._get_output_path()
        self.save_interval = config.save_interval
    
    def _get_output_path(self) -> Path:
        """Generate output file path based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"predictions_"
            f"{self.config.model_config.model_type}_"
            f"{'reasoning' if self.config.model_config.use_reasoning else 'no_reasoning'}_"
            f"{self.config.dataset_config.instruction_type.value}_"
            f"{timestamp}.jsonl"
        )
        return self.config.output_dir / filename
    
    def run(self):
        """Run evaluation on all CSV rows."""
        rows = self.data_loader.get_rows()
        total_rows = len(rows)
        
        logger.info(f"Starting evaluation on {total_rows} rows")
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for idx, row in enumerate(rows, 1):
                    prediction = self._process_row(row, step_num=idx, total_rows=total_rows)
                    json.dump(prediction, f, ensure_ascii=False)
                    f.write("\n")
                    
                    if idx % self.save_interval == 0:
                        f.flush()
                    
                    if idx % 100 == 0:
                        logger.info(f"Processed {idx}/{total_rows} rows ({idx/total_rows*100:.1f}%)")
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            raise e

        logger.info(f"Evaluation completed. Processed {total_rows} rows")
    
    def _process_row(self, row: Dict, step_num: int, total_rows: int) -> Dict:
        """Process a single CSV row and return prediction."""
        logger.info("=" * 80)
        logger.info(f"Step {step_num}/{total_rows} ({step_num/total_rows*100:.1f}%)")
        logger.info("=" * 80)
        
        instruction = row["instruction"]
        image_path = self.data_loader.screenshots_base_dir / row["image_path"]
        
        # Prepare metadata for logging
        metadata = {
            "task_id": row.get("task_id"),
            "step_index": row.get("step_index"),
            "variant": row.get("variant"),
        }
        
        raw_prediction = self.model_client.predict(instruction, image_path, metadata=metadata)
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Image path: {image_path}")
        
        # Truncate very long predictions in logs (likely model hallucination)
        if len(raw_prediction) > 500:
            logger.warning(f"Raw prediction is unusually long ({len(raw_prediction)} chars), truncating log output")
            logger.info(f"Raw prediction (first 500 chars): \n{raw_prediction[:500]}...")
        else:
            logger.info(f"Raw prediction: \n{raw_prediction}")
        
        logger.info(f"Ground truth bbox: {row['target_bounding_box']}")
        logger.info("=" * 80)
        
        return {
            "model": self.config.model_config.model_type,
            "use_reasoning": self.config.model_config.use_reasoning,
            "query_type": self.config.dataset_config.instruction_type.value,
            "test_split": row['split'],
            "variant": metadata["variant"],
            "task_id": row["task_id"],
            "step_index": row["step_index"],
            "instruction": instruction,
            "raw_prediction": raw_prediction,
            "ground_truth_bbox": row["target_bounding_box"],
            "image_path": str(image_path),
        }


# ============================================================================
# Predefined Configurations
# ============================================================================

@dataclass
class EvaluationPreset:
    """Predefined evaluation configuration preset."""
    config_id: str
    model_type: str
    model_name: str  # for vLLM server
    use_reasoning: bool
    instruction_type: InstructionType


def _create_preset(
    model_name: str,
    model_type: str,
    use_reasoning: bool,
    instruction_type: InstructionType,
) -> EvaluationPreset:
    """Create a single preset with explicit config_id."""
    reasoning_str = "reasoning" if use_reasoning else "no_reasoning"
    config_id = f"{model_type}_{reasoning_str}_{instruction_type.value}"
    
    return EvaluationPreset(
        config_id=config_id,
        model_name=model_name,
        model_type=model_type,
        use_reasoning=use_reasoning,
        instruction_type=instruction_type,
    )


def _generate_all_presets() -> Dict[str, EvaluationPreset]:
    """Generate all possible evaluation configuration presets.
    
    Generates 12 total combinations:
    - 3 models (gta1, qwen25vl, uitars15)
    - 2 reasoning modes (with/without)
    - 2 instruction types (direct_query, relational_query)
    
    Note: dataset_variant is specified separately via CLI argument.
    This explicit structure makes it easy to verify correctness and debug.
    """
    presets = {}
    
    # Define all model types explicitly
    MODEL_TYPES = ["gta1", "qwen25vl", "uitars15"]
    
    # Define all other dimensions explicitly
    REASONING_MODES = [False, True]
    MODEL_NAMES = {
        "gta1": "HelloKKMe/GTA1-7B",
        "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "uitars15": "ByteDance-Seed/UI-TARS-1.5-7B",
    }
    INSTRUCTION_TYPES = [
        InstructionType.DIRECT_QUERY,
        InstructionType.RELATIONAL_QUERY,
    ]
    
    # Generate all combinations (without dataset_variant)
    for model_type in MODEL_TYPES:
        for use_reasoning in REASONING_MODES:
            for instruction_type in INSTRUCTION_TYPES:
                preset = _create_preset(
                    model_name=MODEL_NAMES[model_type],
                    model_type=model_type,
                    use_reasoning=use_reasoning,
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
    parser.add_argument("--config_id", type=str, default=None, help="Preset configuration ID (e.g., 'gta1_no_reasoning_direct_query')")
    parser.add_argument("--list_presets", action="store_true", help="List all available preset configuration IDs and exit")
    parser.add_argument("--dataset_variant", default=None, type=str, choices=["style", "precision", "text_zoom", "original"], help="Dataset variant to evaluate")
    
    # Model configuration (optional overrides)
    parser.add_argument("--model_name", type=str, default='ByteDance-Seed/UI-TARS-1.5-7B', help="HuggingFace model identifier for vLLM (e.g., 'ByteDance-Seed/UI-TARS-1.5-7B')")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
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
                  f"Instruction: {preset.instruction_type.value}")
        exit(0)
    
    if args.config_id is None:
        raise ValueError("--config_id is required. Use --list_presets to see available options.")
    
    preset = get_preset(args.config_id)

    # Validate preset model_type
    if preset.model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type '{preset.model_type}' in preset '{args.config_id}'. "
            f"Valid types: {VALID_MODEL_TYPES}"
        )
    
    # Parse dataset variant from CLI argument
    try:
        if args.dataset_variant is not None:
            dataset_variant = DatasetVariantType(args.dataset_variant)
        else:
            dataset_variant = None
    except ValueError:
        raise ValueError(
            f"Invalid dataset_variant: {args.dataset_variant}. "
            f"Must be one of: {[v.value for v in DatasetVariantType]}"
        )
    
    # Log preset details for debugging
    logger.info(f"Using preset: {args.config_id}")
    logger.info(f"  model_type: {preset.model_type}")
    logger.info(f"  use_reasoning: {preset.use_reasoning}")
    logger.info(f"  dataset_variant: {dataset_variant.value if dataset_variant is not None else 'None'}")
    logger.info(f"  instruction_type: {preset.instruction_type.value}")
    
    # Set model-specific default max_tokens if using default value
    # GTA1 only needs ~32 tokens for coordinate output (x,y), but we use 64 to be safe
    if args.max_tokens == 1000:  # Using default value
        max_tokens = DEFAULT_MAX_TOKENS.get((preset.model_type, preset.use_reasoning), args.max_tokens)
    else:
        max_tokens = args.max_tokens  # User explicitly set a value
    
    logger.info(f"  max_tokens: {max_tokens}")
    
    model_config = ModelConfig(
        name=preset.model_name,
        model_type=preset.model_type,
        use_reasoning=preset.use_reasoning,
        temperature=args.temperature,
        max_tokens=max_tokens,
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
        dataset_variant=dataset_variant,
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
    --config_id gta1_no_reasoning_direct_query
"""


if __name__ == "__main__":
    main()
