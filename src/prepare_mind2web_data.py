import json
import os
import sys
import shutil
import math
from PIL import Image
import pandas as pd

GUIPERTURB_TRAINING_SAMPLE_LIST_PATH = '/mnt/disks/sca-data/filtered_gui_dataset_100k.json'
OUTPUT_DATA_DIR = '/mnt/disks/sca-data/processed_training_data'
OUTPUT_SCREENSHOTS_DIR = os.path.join(OUTPUT_DATA_DIR, "screenshots")

def load_training_dataframe() -> pd.DataFrame:
    """Load the training dataframe from the JSON file."""
    return pd.read_json(GUIPERTURB_TRAINING_SAMPLE_LIST_PATH)


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
    op = row['op'].lower()
    type_action_value = row.get('type_action_value', '')
    
    if op in ['click', 'hover', 'click (fake)']:
        # Use click_x and click_y from row, or coordinates if available
        if 'click_x' in row and 'click_y' in row:
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
        if type_action_value is None:
            raise ValueError(f"Missing type_action_value for type action in row")
        ground_truth_action = f"Action: type(content='{type_action_value}')"
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

def process_sample_row(row: pd.Series, parent_directory: str, output_screenshots_dir: str) -> dict | None:
    """
    Process a single row from the dataframe to create a training entry.
    Args:
        row: DataFrame row with sample data
        parent_directory: Root directory containing run folders
        output_screenshots_dir: Directory to copy screenshots to
    Returns:
        Training entry dict or None if processing fails
    """
    # Extract path components
    screenshot_path = row['screenshot']
    run_folder, episode_id, step_index = extract_path_components(screenshot_path)
    
    # Construct paths
    episode_dir = os.path.join(parent_directory, run_folder, episode_id)
    screenshots_dir = os.path.join(episode_dir, "screenshots")
    
    # Find the actual screenshot file
    screenshot_filename = find_screenshot_file(screenshots_dir, step_index)
    if screenshot_filename is None:
        return None
    
    src_screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
    if not os.path.exists(src_screenshot_path):
        return None
    
    # Copy screenshot to output directory
    new_filename = f"{episode_id}_{screenshot_filename}"
    dst_screenshot_path = os.path.join(output_screenshots_dir, new_filename)
    
    try:
        shutil.copy2(src_screenshot_path, dst_screenshot_path)
    except Exception as e:
        print(f"  Warning: Failed to copy {src_screenshot_path}: {e}")
        return None
    
    # Validate required fields
    step_instruction = row.get('step_instruction')
    op = row.get('op')
    
    if step_instruction is None or op is None:
        return None
    
    # Generate prompt and prediction
    prompt = f"<image>\n{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=step_instruction)}"
    
    try:
        prediction = generate_action_prediction(row, src_screenshot_path)
    except Exception as e:
        print(f"  Warning: Failed to generate action for {episode_id} step {step_index}: {e}")
        return None
    
    # Create training entry
    entry = {
        "id": f"{run_folder}_{episode_id}_step_{step_index}",
        "image": [os.path.join("screenshots", new_filename)],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": prediction}
        ]
    }
    
    return entry


def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: python prepare_mind2web_data.py <parent_directory> [max_samples]")
        print("Example: python prepare_mind2web_data.py /mnt/disks/sca-data/all_training_splits")
        print("Example: python prepare_mind2web_data.py /mnt/disks/sca-data/all_training_splits 1000")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    
    # Optional parameter to limit number of samples processed
    max_samples = None
    if len(sys.argv) >= 3:
        try:
            max_samples = int(sys.argv[2])
            if max_samples <= 0:
                print("Error: max_samples must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("Error: max_samples must be a valid integer")
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
    
    # Shuffle dataframe to randomize order (avoid episode-based ordering)
    print("Shuffling dataframe...")
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    print("Dataframe shuffled")
    
    # Limit samples if max_samples is specified
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Create output directories
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SCREENSHOTS_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DATA_DIR, "training_data.json")

    print(f"\nOutput directory: {OUTPUT_DATA_DIR}")
    print(f"Screenshots will be copied to: {OUTPUT_SCREENSHOTS_DIR}")
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
    
    print(f"\nProcessing {len(df)} samples...")
    for idx, row in df.iterrows():
        # Extract entry ID to check if already processed
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
        # Process the row
        entry = process_sample_row(row, parent_directory, OUTPUT_SCREENSHOTS_DIR)
        
        if entry is not None:
            all_training_data.append(entry)
            existing_entry_ids.add(entry_id)  # Add to set to avoid duplicates
            processed_count += 1
            
            # Save incrementally after each successful processing
            try:
                with open(output_file_path, 'w') as f:
                    json.dump(all_training_data, f, indent=4)
            except Exception as e:
                print(f"  Warning: Failed to save incrementally: {e}")
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples ({processed_count} new, {already_exists_count} existing, {skipped_count} failed)...")
        else:
            skipped_count += 1
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples ({processed_count} new, {already_exists_count} existing, {skipped_count} failed)...")
    
    print(f"\n{'='*60}")
    print(f"Total training examples: {len(all_training_data)}")
    print(f"Newly processed: {processed_count}")
    print(f"Already existed: {already_exists_count}")
    print(f"Skipped (failed): {skipped_count}")
    
    print(f"\nSuccessfully wrote training data to {output_file_path}")
    print(f"Total screenshots in output directory: {len(all_training_data)}")


if __name__ == "__main__":
    main()