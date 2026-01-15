import json
import os
import sys
import shutil
import math
import re
from collections import Counter
from PIL import Image
import pandas as pd

GUIPERTURB_TRAINING_SAMPLE_LIST_PATH = '/mnt/disks/sca-data/filtered_gui_dataset_100k.json'
CSV_TRAINING_SAMPLE_LIST_PATH = '/Users/lockewang/Downloads/variant_data_cleaned.csv'
# OUTPUT_DATA_DIR = '/mnt/disks/sca-data/processed_training_data'
OUTPUT_DATA_DIR = '/Users/lockewang/Downloads/test_output'

def filter_style_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for style variant: run folders ending with *_style or *_train, or using variant column."""
    # Check if CSV format with variant column
    if 'variant' in df.columns:
        return df[df['variant'].isin(['style', 'original'])].reset_index(drop=True)
    
    # Fallback to original folder-based filtering for JSON format
    def is_style_run_folder(screenshot_path: str) -> bool:
        try:
            run_folder, _, _ = extract_path_components(screenshot_path)
            return run_folder.endswith('_style') or run_folder.endswith('_train')
        except:
            return False
    
    mask = df['screenshot'].apply(is_style_run_folder)
    return df[mask].reset_index(drop=True)


def filter_text_shrink_zoom_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for text_shrink/zoom variant: run folders ending with *_text_shrink or *_precision, or using variant column."""
    # Check if CSV format with variant column
    if 'variant' in df.columns:
        return df[df['variant'].isin(['text_shrink', 'precision'])].reset_index(drop=True)
    
    # Fallback to original folder-based filtering for JSON format
    def is_text_shrink_zoom_run_folder(screenshot_path: str) -> bool:
        try:
            run_folder, _, _ = extract_path_components(screenshot_path)
            return run_folder.endswith('_text_shrink') or run_folder.endswith('_precision')
        except:
            return False
    
    mask = df['screenshot'].apply(is_text_shrink_zoom_run_folder)
    return df[mask].reset_index(drop=True)


def create_balanced_val_set(df: pd.DataFrame, val_ratio: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced validation set by randomly selecting samples evenly from each run folder.
    Args:
        df: Input dataframe
        val_ratio: Ratio of samples to select for validation (default 0.1 = 10%)
        random_state: Random seed for reproducibility
    Returns:
        Filtered dataframe with balanced validation samples
    """
    # Check if CSV format
    is_csv_format = 'image_path' in df.columns
    
    def get_run_folder(row_or_path):
        try:
            if is_csv_format:
                run_folder, _, _ = extract_path_components_from_csv(row_or_path)
            else:
                run_folder, _, _ = extract_path_components(row_or_path)
            return run_folder
        except:
            return None
    
    df_with_run_folder = df.copy()
    if is_csv_format:
        df_with_run_folder['run_folder'] = df_with_run_folder.apply(get_run_folder, axis=1)
    else:
        df_with_run_folder['run_folder'] = df_with_run_folder['screenshot'].apply(get_run_folder)
    
    # Group by run folder
    val_samples = []
    for run_folder, group in df_with_run_folder.groupby('run_folder'):
        n_samples = max(1, int(len(group) * val_ratio))
        val_group = group.sample(n=n_samples, random_state=random_state)
        val_samples.append(val_group)
    
    val_df = pd.concat(val_samples, ignore_index=True)
    return val_df.drop(columns=['run_folder']).reset_index(drop=True)


def load_training_dataframe(data_path: str = None) -> pd.DataFrame:
    """Load the training dataframe from JSON or CSV file."""
    if data_path is None:
        # Try CSV first, fallback to JSON
        if os.path.exists(CSV_TRAINING_SAMPLE_LIST_PATH):
            data_path = CSV_TRAINING_SAMPLE_LIST_PATH
        else:
            data_path = GUIPERTURB_TRAINING_SAMPLE_LIST_PATH
    
    # Determine format by extension
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    else:
        return pd.read_json(data_path)


def extract_path_components(screenshot_path: str) -> tuple[str, str, int]:
    """
    Extract run_folder, episode_id, and step_index from screenshot path.
    Args:
        screenshot_path: Full path to screenshot (e.g., '/mnt/disks/sca-data/all_training_splits/run_20251126_013744_train/d070774f-9ca2-43c0-a7d0-221697791cf0/screenshots/step_1_click.png')
    Returns:
        (run_folder, episode_id, step_index): Tuple of extracted components
    """
    parts = [p for p in screenshot_path.split('/') if p]  # Remove empty strings
    filename = parts[-1]
    
    # Find screenshots directory index in the filepath
    screenshots_idx = None
    for i, part in enumerate(parts):
        if part == 'screenshots':
            screenshots_idx = i
            break
    
    if screenshots_idx is None:
        raise ValueError(f"Could not find 'screenshots' directory in path: {screenshot_path}")
    
    # episode_id is the directory before screenshots
    episode_id = parts[screenshots_idx - 1]
    # run_folder is the directory before episode_id
    run_folder = parts[screenshots_idx - 2]
    
    # Extract step_index from filename (format: step_<index>_<action>.png)
    step_index = int(filename.split('_')[1])
    
    return run_folder, episode_id, step_index


def parse_target_coordinates(coord_str: str) -> tuple[int, int]:
    """Parse target_coordinates string '(x, y)' to (x, y) tuple."""
    match = re.match(r'\((\d+),\s*(\d+)\)', coord_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid coordinate format: {coord_str}")


def extract_op_from_image_path(image_path: str) -> str:
    """Extract action type from image filename (e.g., step_3_click.png -> 'click')."""
    filename = os.path.basename(image_path)
    match = re.search(r'step_\d+_(\w+)\.png', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract action type from: {image_path}")


def extract_type_value_from_instruction(instruction: str) -> str | None:
    """Extract type content from instruction (e.g., 'Type 'pork'...' -> 'pork'). Returns None if no value found."""
    # Try pattern: Type 'value' or Type "value"
    patterns = [
        r"Type\s+['\"]([^'\"]+)['\"]",  # Matches both single and double quotes
        r"Type\s+['\"]([^'\"]*?)['\"]",  # Non-greedy version
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return match.group(1)
    return None  # No value found - instruction is just "Type the textbox above X"


def extract_path_components_from_csv(row: pd.Series) -> tuple[str, str, int]:
    """
    Extract run_folder, episode_id, and step_index from CSV row.
    Args:
        row: DataFrame row with CSV data
    Returns:
        (run_folder, episode_id, step_index): Tuple of extracted components
    """
    image_path = row['image_path']
    parts = [p for p in image_path.split('/') if p]
    
    # Find screenshots directory index
    screenshots_idx = None
    for i, part in enumerate(parts):
        if part == 'screenshots':
            screenshots_idx = i
            break
    
    if screenshots_idx is None:
        raise ValueError(f"Could not find 'screenshots' directory in path: {image_path}")
    
    # episode_id (task_id) is the directory before screenshots
    episode_id = parts[screenshots_idx - 1]
    # run_folder is the directory before episode_id
    run_folder = parts[screenshots_idx - 2]
    # step_index is from the column
    step_index = int(row['step_index'])
    
    return run_folder, episode_id, step_index


def find_screenshot_file(screenshots_dir: str, step_index: int) -> str | None:
    """
    Find the screenshot file matching step_index pattern.
    Args:
        screenshots_dir: Directory containing screenshots
        step_index: Step index to match
    Returns:
        Filename if found, None otherwise
    """
    if not os.path.exists(screenshots_dir):
        return None
    
    pattern = f"step_{step_index}_"
    for filename in os.listdir(screenshots_dir):
        if filename.startswith(pattern) and filename.endswith('.png'):
            return filename
    return None

#The following is borrwed from the UI Tars Codebase
UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}
"""

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def get_image_dimensions(image_path: str) -> tuple[int, int]:
    """
    Get the width and height of an image from its file path.
    Args:
        image_path: Path to the image file
    Returns:
        (width, height): Tuple containing the image width and height in pixels
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")


def prepare_training_coordinates(original_x, original_y, original_width, original_height):
    """
    Convert original image coordinates to smart-resized space for training.
    Args:
        original_x, original_y: Click position in original image
        original_width, original_height: Original image dimensions
    Returns:
        training_x, training_y: Coordinates in smart-resized space
    """
    # Get smart-resized dimensions
    smart_h, smart_w = smart_resize(
        height=original_height,
        width=original_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )

    # Scale coordinates
    training_x = int(original_x * smart_w / original_width)
    training_y = int(original_y * smart_h / original_height)

    return (training_x, training_y)


def generate_action_prediction(row: pd.Series, screenshot_path: str) -> str:
    """
    Generate action prediction string from row data.
    Args:
        row: DataFrame row with action data
        screenshot_path: Path to screenshot for coordinate normalization
    Returns:
        Action prediction string
    """
    # Extract op from filename if not in row (CSV format)
    if 'op' in row and pd.notna(row['op']):
        op = row['op'].lower()
    elif 'image_path' in row:
        op = extract_op_from_image_path(row['image_path']).lower()
    else:
        raise ValueError(f"Could not determine action type from row")
    
    if op in ['click', 'hover', 'click (fake)', 'select']:
        # Parse target_coordinates from CSV format "(x, y)" or use JSON format
        if 'target_coordinates' in row and pd.notna(row['target_coordinates']):
            click_x, click_y = parse_target_coordinates(row['target_coordinates'])
        elif 'click_x' in row and 'click_y' in row:
            click_x, click_y = row['click_x'], row['click_y']
        elif 'coordinates' in row and isinstance(row['coordinates'], (list, tuple)) and len(row['coordinates']) >= 2:
            click_x, click_y = row['coordinates'][0], row['coordinates'][1]
        else:
            raise ValueError(f"Missing coordinates for click action in row")
        
        original_width, original_height = get_image_dimensions(screenshot_path)
        normalized_coordinates = prepare_training_coordinates(click_x, click_y, original_width, original_height)
        ground_truth_action = f"Action: click(start_box='({normalized_coordinates[0]}, {normalized_coordinates[1]})')"
        print(f"Ground truth action: {ground_truth_action}")
        return ground_truth_action
    elif op == "type":
        # Extract coordinates for click action (same logic as click actions)
        if 'target_coordinates' in row and pd.notna(row['target_coordinates']):
            click_x, click_y = parse_target_coordinates(row['target_coordinates'])
        elif 'click_x' in row and 'click_y' in row:
            click_x, click_y = row['click_x'], row['click_y']
        elif 'coordinates' in row and isinstance(row['coordinates'], (list, tuple)) and len(row['coordinates']) >= 2:
            click_x, click_y = row['coordinates'][0], row['coordinates'][1]
        else:
            raise ValueError(f"Missing coordinates for type action in row")
        
        # Normalize coordinates
        original_width, original_height = get_image_dimensions(screenshot_path)
        normalized_coordinates = prepare_training_coordinates(click_x, click_y, original_width, original_height)
        
        # Extract type value from instruction if not in row
        # CSV format: try multi_element_instruction first, then step_instruction
        # JSON format: use step_instruction (backward compatible)
        if 'type_action_value' in row and pd.notna(row['type_action_value']):
            type_action_value = row['type_action_value']
        elif 'multi_element_instruction' in row and pd.notna(row['multi_element_instruction']):
            type_action_value = extract_type_value_from_instruction(row['multi_element_instruction'])
        elif 'step_instruction' in row:
            type_action_value = extract_type_value_from_instruction(row['step_instruction'])
        else:
            type_action_value = None
        
        # If no type value found, treat as click action instead (e.g., "Type the textbox above X" = just click)
        # This handles both CSV and JSON formats
        # Note: normalized_coordinates already calculated above, so we can reuse them
        if type_action_value is None or type_action_value == '':
            # Fall back to click action - this handles cases like "Type the textbox above X" or "Click on the textbox"
            ground_truth_action = f"Action: click(start_box='({normalized_coordinates[0]}, {normalized_coordinates[1]})')"
            print(f"Ground truth action (type->click fallback): {ground_truth_action}")
            return ground_truth_action
        
        # Generate combined action: click first, then type, separated by \n\n
        click_action = f"Action: click(start_box='({normalized_coordinates[0]}, {normalized_coordinates[1]})')"
        type_action = f"type(content='{type_action_value}')"
        ground_truth_action = f"{click_action}\n\n{type_action}"
        print(f"Ground truth action: {ground_truth_action}")
        return ground_truth_action
    elif op == "press enter":
        ground_truth_action = f"Action: type(content='\\n')"
        print(f"Ground truth action: {ground_truth_action}")
        return ground_truth_action
    elif op == "ignore":
        ground_truth_action = f"Action: wait()"
        print(f"Ground truth action: {ground_truth_action}")
        return ground_truth_action
    else:
        raise ValueError(f"Unknown action type: {op}")

def process_sample_row(row: pd.Series, parent_directory: str, output_screenshots_dir: str) -> tuple[dict | None, str | None]:
    """
    Process a single row from the dataframe to create a training entry.
    Supports both JSON and CSV formats.
    Args:
        row: DataFrame row with sample data
        parent_directory: Root directory containing run folders
        output_screenshots_dir: Directory to copy screenshots to
    Returns:
        Tuple of (Training entry dict or None if processing fails, error reason string or None)
    """
    # Check if CSV format (has image_path) or JSON format (has screenshot)
    is_csv_format = 'image_path' in row
    
    if is_csv_format:
        # CSV format: extract components and search by step_index pattern (robust)
        image_path = row['image_path']  # Keep for reference, but don't use for file lookup
        run_folder, episode_id, step_index = extract_path_components_from_csv(row)
        
        # Construct directory path (same as JSON format)
        episode_dir = os.path.join(parent_directory, run_folder, episode_id)
        screenshots_dir = os.path.join(episode_dir, "screenshots")
        
        # Find the actual screenshot file by step_index pattern (robust)
        screenshot_filename = find_screenshot_file(screenshots_dir, step_index)
        if screenshot_filename is None:
            return None, f"Screenshot file not found in directory: {screenshots_dir} (step_index: {step_index})"
        
        src_screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
        if not os.path.exists(src_screenshot_path):
            return None, f"Screenshot file not found: {src_screenshot_path}"
    else:
        # JSON format: screenshot is absolute path, extract components from path
        screenshot_path = row['screenshot']
        run_folder, episode_id, step_index = extract_path_components(screenshot_path)
        
        # Construct paths
        episode_dir = os.path.join(parent_directory, run_folder, episode_id)
        screenshots_dir = os.path.join(episode_dir, "screenshots")
        
        # Find the actual screenshot file
        screenshot_filename = find_screenshot_file(screenshots_dir, step_index)
        if screenshot_filename is None:
            return None, f"Screenshot file not found in directory: {screenshots_dir} (step_index: {step_index})"
        
        src_screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
        if not os.path.exists(src_screenshot_path):
            return None, f"Screenshot file not found: {src_screenshot_path}"
    
    # Copy screenshot to output directory
    new_filename = f"{episode_id}_{screenshot_filename}"
    dst_screenshot_path = os.path.join(output_screenshots_dir, new_filename)
    
    try:
        shutil.copy(src_screenshot_path, dst_screenshot_path)
    except Exception as e:
        error_msg = f"Failed to copy screenshot: {src_screenshot_path} -> {dst_screenshot_path}: {e}"
        print(f"  Warning: {error_msg}")
        return None, error_msg
    
    # Validate required fields - use multi_element_instruction for CSV, step_instruction for JSON
    if is_csv_format:
        instruction = row.get('multi_element_instruction')
        if instruction is None:
            return None, "Missing multi_element_instruction field"
    else:
        instruction = row.get('step_instruction')
        if instruction is None:
            return None, "Missing step_instruction field"
    
    # Extract op from filename if not in row (CSV format)
    if 'op' in row and pd.notna(row['op']):
        op = row['op']
    elif is_csv_format:
        try:
            op = extract_op_from_image_path(image_path)
        except Exception as e:
            return None, f"Failed to extract op from image_path: {e}"
    else:
        op = row.get('op')
        if op is None:
            return None, "Missing op field"
    
    # Generate prompt and prediction
    prompt = f"<image>\n{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=instruction)}"
    
    try:
        prediction = generate_action_prediction(row, src_screenshot_path)
    except Exception as e:
        error_msg = f"Failed to generate action for {episode_id} step {step_index}: {e}"
        print(f"  Warning: {error_msg}")
        return None, error_msg
    
    # Create training entry
    entry = {
        "id": f"{run_folder}_{episode_id}_step_{step_index}",
        "image": [os.path.join("screenshots", new_filename)],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": prediction}
        ]
    }
    
    return entry, None


def main():
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python prepare_mind2web_data.py <parent_directory> [variant] [max_samples]")
        print("Variants:")
        print("  (default) - All samples")
        print("  style - Style variant (run folders ending with *_style or *_train, or variant='style'/'original' in CSV)")
        print("  text_shrink_zoom - Text shrink/zoom variant (run folders ending with *_text_shrink or *_precision, or variant='text_shrink'/'precision' in CSV)")
        print("  val - Balanced validation set (10% evenly from each run folder)")
        print("\nData formats supported:")
        print("  - JSON format: expects 'screenshot', 'op', 'step_instruction', 'click_x', 'click_y' columns")
        print("  - CSV format: expects 'image_path', 'step_index', 'task_id', 'variant', 'multi_element_instruction', 'target_coordinates' columns")
        print("    CSV format auto-detected if CSV_TRAINING_SAMPLE_LIST_PATH exists, otherwise uses JSON")
        print("\nExamples:")
        print("  python prepare_mind2web_data.py /mnt/disks/sca-data/all_training_splits")
        print("  python prepare_mind2web_data.py /mnt/disks/sca-data/all_training_splits style")
        print("  python prepare_mind2web_data.py /mnt/disks/sca-data/all_training_splits val 1000")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    
    # Parse variant (optional)
    variant = None
    max_samples = None
    
    if len(sys.argv) >= 3:
        arg2 = sys.argv[2]
        if arg2 in ['style', 'text_shrink_zoom', 'val', 'all']:
            variant = arg2
            # Check for max_samples as 3rd argument
            if len(sys.argv) >= 4:
                try:
                    max_samples = int(sys.argv[3])
                    if max_samples <= 0:
                        print("Error: max_samples must be a positive integer")
                        sys.exit(1)
                except ValueError:
                    print("Error: max_samples must be a valid integer")
                    sys.exit(1)
        else:
            # Assume it's max_samples (backward compatibility)
            try:
                max_samples = int(arg2)
                if max_samples <= 0:
                    print("Error: max_samples must be a positive integer")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Unknown variant '{arg2}'. Must be 'style', 'text_shrink_zoom', 'val', or 'all'")
                sys.exit(1)
    
    # Validate parent directory exists
    if not os.path.exists(parent_directory):
        print(f"Error: Directory {parent_directory} does not exist")
        sys.exit(1)
    
    if not os.path.isdir(parent_directory):
        print(f"Error: {parent_directory} is not a directory")
        sys.exit(1)
    
    # Load training dataframe
    print("Loading training dataframe...")
    df = load_training_dataframe()
    print(f"Loaded {len(df)} samples from dataframe")
    
    # Apply variant filtering
    if variant == 'style':
        print("\nFiltering for style variant (run folders ending with *_style or *_train)...")
        df = filter_style_variant(df)
        print(f"Filtered to {len(df)} samples for style variant")
    elif variant == 'text_shrink_zoom':
        print("\nFiltering for text_shrink/zoom variant (run folders ending with *_text_shrink or *_precision)...")
        df = filter_text_shrink_zoom_variant(df)
        print(f"Filtered to {len(df)} samples for text_shrink/zoom variant")
    elif variant == 'val':
        print("\nCreating balanced validation set (10% evenly from each run folder)...")
        df = create_balanced_val_set(df, val_ratio=0.1, random_state=42)
        print(f"Created validation set with {len(df)} samples")
    elif variant == 'all':
        print("\nNo filtering for all variant")
    
    # Shuffle dataframe to randomize order (avoid episode-based ordering)
    print("Shuffling dataframe...")
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    print("Dataframe shuffled")
    
    # Limit samples if max_samples is specified
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Determine output paths based on variant
    if variant:
        output_data_dir = os.path.join(OUTPUT_DATA_DIR, variant)
        output_screenshots_dir = os.path.join(output_data_dir, "screenshots")
        output_file_path = os.path.join(output_data_dir, f"{variant}_data.json")
    else:
        raise ValueError("Variant is required")

    # Create output directories
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_screenshots_dir, exist_ok=True)

    print(f"\nOutput directory: {output_data_dir}")
    print(f"Screenshots will be copied to: {output_screenshots_dir}")
    print(f"Training data will be saved to: {output_file_path}")
    
    # Load existing training data if it exists
    all_training_data = []
    existing_entry_ids = set()
    
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r') as f:
                all_training_data = json.load(f)
            existing_entry_ids = {entry['id'] for entry in all_training_data}
            print(f"Loaded {len(all_training_data)} existing entries from {output_file_path}")
        except Exception as e:
            print(f"Warning: Failed to load existing training data: {e}. Starting fresh.")
            all_training_data = []
            existing_entry_ids = set()
    
    # Process each row in dataframe
    processed_count = 0
    skipped_count = 0
    already_exists_count = 0
    skipped_samples = []  # Track skipped samples with reasons
    
    # Check if CSV format
    is_csv_format = 'image_path' in df.columns
    
    print(f"\nProcessing {len(df)} samples...")
    for idx, row in df.iterrows():
        # Extract entry ID to check if already processed
        if is_csv_format:
            run_folder, episode_id, step_index = extract_path_components_from_csv(row)
        else:
            screenshot_path = row['screenshot']
            run_folder, episode_id, step_index = extract_path_components(screenshot_path)
        entry_id = f"{run_folder}_{episode_id}_step_{step_index}"
        
        # Skip if already processed
        if entry_id in existing_entry_ids:
            already_exists_count += 1
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples (skipped {already_exists_count} existing)...")
            continue
        
        print(f"Processing sample {idx + 1}/{len(df)}: {entry_id}")
        # Process the row (will overwrite image if it exists to prevent corruption)
        entry, error_reason = process_sample_row(row, parent_directory, output_screenshots_dir)
        
        if entry is not None:
            all_training_data.append(entry)
            existing_entry_ids.add(entry_id)  # Add to set to avoid duplicates
            processed_count += 1
            
            # Save incrementally every 100 files (much faster than saving after each file)
            if processed_count % 100 == 0:
                try:
                    with open(output_file_path, 'w') as f:
                        json.dump(all_training_data, f, indent=4)
                except Exception as e:
                    print(f"  Warning: Failed to save incrementally: {e}")
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples ({processed_count} new, {already_exists_count} existing, {skipped_count} failed)...")
        else:
            skipped_count += 1
            skipped_samples.append({
                'entry_id': entry_id,
                'reason': error_reason or 'Unknown error',
                'row_index': idx
            })
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples ({processed_count} new, {already_exists_count} existing, {skipped_count} failed)...")
    
    # Final save to ensure all data is written
    try:
        with open(output_file_path, 'w') as f:
            json.dump(all_training_data, f, indent=4)
    except Exception as e:
        print(f"  Warning: Failed to save final training data: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total training examples: {len(all_training_data)}")
    print(f"Newly processed: {processed_count}")
    print(f"Already existed: {already_exists_count}")
    print(f"Skipped (failed): {skipped_count}")
    
    # Print skipped samples for debugging
    if skipped_count > 0:
        print(f"\n{'='*60}")
        print(f"SKIPPED SAMPLES DETAILS ({skipped_count} total):")
        print(f"{'='*60}")
        for i, skipped in enumerate(skipped_samples, 1):
            print(f"\n{i}. Entry ID: {skipped['entry_id']}")
            print(f"   Row Index: {skipped['row_index']}")
            print(f"   Reason: {skipped['reason']}")
        
        # Group by reason for summary
        reason_counts = Counter(s['reason'] for s in skipped_samples)
        print(f"\n{'='*60}")
        print("SKIP REASONS SUMMARY:")
        print(f"{'='*60}")
        for reason, count in reason_counts.most_common():
            print(f"  {count:3d} - {reason}")
        print(f"{'='*60}")
        print("\nPress Enter to continue...")
        input()  # Pause for debugging
    
    print(f"\nSuccessfully wrote training data to {output_file_path}")
    print(f"Total screenshots in output directory: {len(all_training_data)}")


if __name__ == "__main__":
    main()