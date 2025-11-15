import json
import os
import sys
import shutil
from pathlib import Path

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

def process_subdirectory(subdir_path, subdir_name, output_screenshots_dir):
    """Process a single subdirectory containing trajectory.json and screenshots."""
    trajectory_path = os.path.join(subdir_path, "trajectory.json")
    screenshots_path = os.path.join(subdir_path, "screenshots")
    
    # Check if trajectory.json exists
    if not os.path.exists(trajectory_path):
        print(f"  Warning: No trajectory.json found in {subdir_path}, skipping...")
        return None
    
    # Load trajectory data
    try:
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
    except Exception as e:
        print(f"  Error loading {trajectory_path}: {e}, skipping...")
        return None
    
    print(f"  Successfully loaded {len(trajectory_data)} items from {trajectory_path}")
    
    # Get all screenshots in order
    if not os.path.exists(screenshots_path):
        print(f"  Warning: No screenshots directory found in {subdir_path}, skipping...")
        return None
    
    # Get all screenshot files and sort them
    screenshot_files = sorted([f for f in os.listdir(screenshots_path) if f.endswith('.png')])
    
    if not screenshot_files:
        print(f"  Warning: No screenshots found in {screenshots_path}, skipping...")
        return None
    
    # Copy screenshots to output directory with prepended subdirectory name
    image_paths = []
    for screenshot_file in screenshot_files:
        src_path = os.path.join(screenshots_path, screenshot_file)
        # Prepend subdirectory name to make filename unique
        new_filename = f"{subdir_name}_{screenshot_file}"
        dst_path = os.path.join(output_screenshots_dir, new_filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            # Store relative path from data directory
            image_paths.append(os.path.join("screenshots", new_filename))
        except Exception as e:
            print(f"  Warning: Failed to copy {src_path} to {dst_path}: {e}")
            continue
    
    # Build conversations list with one human/gpt pair per trajectory step
    conversations = []
    
    for example in trajectory_data:
        step_instruction = example.get("step_instruction")
        op = example.get("op")
        coordinates = example.get("coordinates")
        type_action_value = example.get("type_action_value")
        
        # Skip if any of these necessary fields is missing
        if step_instruction is None or op is None or coordinates is None:
            continue

        # Format prompt - include <image> token for each image
        image_tokens = "".join(["<image>\n" for _ in image_paths])
        prompt = f"{image_tokens}{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=step_instruction)}"

        # Mind2Web has actions: Click, Type, Hover, Press Enter, Click (Fake) and Ignore. 
        # Map these actions to the UI Tars actions
        prediction = ""
        if op.lower() == "click" or op.lower() == "hover" or op.lower() == "click (fake)":
            prediction = f"Action: click(start_box='({coordinates[0]}, {coordinates[1]})')"
        elif op.lower() == "type":
            prediction = f"Action: type(content='({type_action_value})')"
        elif op.lower() == "press enter":
            prediction = f"Action: type(content='(\\n)')"   
        elif op.lower() == "ignore":
            prediction = f"Action: wait()"
        
        conversations.append({
            "from": "human",
            "value": prompt
        })
        conversations.append({
            "from": "gpt",
            "value": prediction
        })
    
    if not conversations:
        print(f"  Warning: No valid conversations generated for {subdir_path}, skipping...")
        return None
    
    # Create single entry for this subdirectory
    entry = {
        "id": subdir_name,
        "image": image_paths,
        "conversations": conversations
    }
    
    return entry


def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: python prepare_mind2web_data.py <parent_directory> [max_subdirs]")
        print("Example: python prepare_mind2web_data.py /mnt/sca-web-data/run_20251112_004959_test_domain")
        print("Example: python prepare_mind2web_data.py /mnt/sca-web-data/run_20251112_004959_test_domain 10")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    
    # Optional parameter to limit number of subdirectories processed
    max_subdirs = None
    if len(sys.argv) >= 3:
        try:
            max_subdirs = int(sys.argv[2])
            if max_subdirs <= 0:
                print("Error: max_subdirs must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("Error: max_subdirs must be a valid integer")
            sys.exit(1)
    
    # Validate parent directory exists
    if not os.path.exists(parent_directory):
        print(f"Error: Directory {parent_directory} does not exist")
        sys.exit(1)
    
    if not os.path.isdir(parent_directory):
        print(f"Error: {parent_directory} is not a directory")
        sys.exit(1)
    
    print(f"Processing subdirectories in: {parent_directory}")
    
    # Get all subdirectories
    subdirectories = [d for d in os.listdir(parent_directory) 
                     if os.path.isdir(os.path.join(parent_directory, d))]
    
    if not subdirectories:
        print(f"Error: No subdirectories found in {parent_directory}")
        sys.exit(1)
    
    # Limit subdirectories if max_subdirs is specified
    if max_subdirs is not None:
        subdirectories = subdirectories[:max_subdirs]
        print(f"Found {len(subdirectories)} subdirectories to process (limited to {max_subdirs})")
    else:
        print(f"Found {len(subdirectories)} subdirectories to process")
    
    # Create output directories
    current_dir = os.getcwd()
    output_data_dir = os.path.join(current_dir, "data")
    output_screenshots_dir = os.path.join(output_data_dir, "screenshots")
    
    # Create directories if they don't exist
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_screenshots_dir, exist_ok=True)
    print(f"\nOutput directory: {output_data_dir}")
    print(f"Screenshots will be copied to: {output_screenshots_dir}")
    
    # Process each subdirectory
    all_training_data = []
    
    for subdir_name in sorted(subdirectories):
        subdir_path = os.path.join(parent_directory, subdir_name)
        print(f"\nProcessing: {subdir_name}")
        
        entry = process_subdirectory(subdir_path, subdir_name, output_screenshots_dir)
        
        if entry is not None:
            all_training_data.append(entry)
            print(f"  Added training entry with {len(entry['conversations'])//2} steps and {len(entry['image'])} images from {subdir_name}")
        else:
            print(f"  Skipped {subdir_name} due to errors or missing data")
    
    print(f"\n{'='*60}")
    print(f"Total training examples collected: {len(all_training_data)}")
    
    # Define output path in the data directory
    output_file_path = os.path.join(output_data_dir, "training_data.json")
    
    # Write the training_data to the JSON file
    with open(output_file_path, 'w') as f:
        json.dump(all_training_data, f, indent=4)
    
    print(f"Successfully wrote training data to {output_file_path}")
    print(f"Copied {sum(len(entry['image']) for entry in all_training_data)} screenshots to {output_screenshots_dir}")


if __name__ == "__main__":
    main()