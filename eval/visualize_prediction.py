"""
Visualize predicted coordinates on screenshots for debugging.

Usage:
    python -m eval.visualize_predictions \\
        --summary_json uitars_eval_20251117_045004/uitars_summary.json \\
        --episode_id 00eace8c-829d-4f27-a6b3-e05aeaa96881 \\
        --step_index 0

    # Or visualize all steps in an episode
    python -m eval.visualize_predictions \\
        --summary_json uitars_eval_20251117_045004/uitars_summary.json \\
        --episode_id 00eace8c-829d-4f27-a6b3-e05aeaa96881

    # Or visualize a single prediction directly
    python -m eval.visualize_predictions \\
        --image_path /path/to/screenshot.png \\
        --prediction "Action: click(start_box='(739,336)')"
"""
import argparse
import json
import re
import ast
from pathlib import Path
from typing import Optional, Tuple, List

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL (Pillow) package not installed. Install with: pip install Pillow")
    exit(1)


def parse_prediction_coordinates(prediction_text: str) -> List[Tuple[float, float]]:
    """
    Extract coordinates from prediction text.
    
    Examples:
        "Action: click(start_box='(739,336)')" -> [(739, 336)]
        "Action: drag(start_box='(100,200)', end_box='(300,400)')" -> [(100, 200), (300, 400)]
    
    Returns:
        List of (x, y) coordinate tuples
    """
    coordinates = []
    
    # Extract all coordinate pairs from start_box and end_box
    patterns = [
        r"start_box=['\"]([^'\"]+)['\"]",
        r"end_box=['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, prediction_text)
        for match in matches:
            # Try to parse as tuple/list
            try:
                # Remove box markers if present
                coords_str = match.replace("<|box_start|>", "").replace("<|box_end|>", "")
                coords_str = coords_str.strip()
                
                # Try parsing as tuple/list
                parsed = ast.literal_eval(coords_str)
                if isinstance(parsed, (list, tuple)):
                    if len(parsed) >= 2:
                        # Take first two values as (x, y)
                        x, y = float(parsed[0]), float(parsed[1])
                        coordinates.append((x, y))
            except (ValueError, SyntaxError):
                # Try parsing as "(x,y)" format
                coords_str = match.strip()
                if coords_str.startswith("(") and coords_str.endswith(")"):
                    coords_str = coords_str[1:-1]
                    parts = coords_str.split(",")
                    if len(parts) >= 2:
                        x, y = float(parts[0].strip()), float(parts[1].strip())
                        coordinates.append((x, y))
    
    return coordinates


def visualize_prediction(
    image_path: str,
    prediction_text: str,
    ground_truth: Optional[List[str]] = None,
    instruction: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Visualize predicted coordinates on an image.
    
    Args:
        image_path: Path to screenshot image
        prediction_text: Raw prediction text containing coordinates
        ground_truth: Optional list of ground truth UITARS actions
        instruction: Optional instruction text
        output_path: Optional path to save the visualization
    """
    # Load image
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Parse coordinates
    coords = parse_prediction_coordinates(prediction_text)
    
    if not coords:
        print(f"No coordinates found in prediction: {prediction_text}")
        return
    
    # Create a copy for drawing
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Draw predicted points
    point_radius = 10
    for i, (x, y) in enumerate(coords):
        # Convert to int if needed
        x, y = int(round(x)), int(round(y))
        
        # Draw circle for predicted point
        color = "red" if i == 0 else "orange"  # First point red, others orange
        bbox = [
            x - point_radius,
            y - point_radius,
            x + point_radius,
            y + point_radius,
        ]
        draw.ellipse(bbox, fill=color, outline="darkred", width=2)
        
        # Draw crosshair
        crosshair_size = 20
        draw.line(
            [(x - crosshair_size, y), (x + crosshair_size, y)],
            fill=color,
            width=2,
        )
        draw.line(
            [(x, y - crosshair_size), (x, y + crosshair_size)],
            fill=color,
            width=2,
        )
        
        # Label
        label = f"Pred {i+1}" if len(coords) > 1 else "Predicted"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw label background
        text_bbox = draw.textbbox((x + point_radius + 5, y - 10), label, font=font)
        draw.rectangle(text_bbox, fill="white", outline=color, width=1)
        draw.text((x + point_radius + 5, y - 10), label, fill=color, font=font)
    
    # Add text overlay with prediction info
    overlay_text = []
    if instruction:
        overlay_text.append(f"Instruction: {instruction[:100]}...")
    overlay_text.append(f"Prediction: {prediction_text[:150]}...")
    if ground_truth:
        overlay_text.append(f"Ground Truth: {', '.join(ground_truth[:2])}...")
    overlay_text.append(f"Coordinates: {coords}")
    
    # Draw text overlay at top
    y_offset = 10
    for text_line in overlay_text:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Get text size and draw background
        text_bbox = draw.textbbox((10, y_offset), text_line, font=font)
        padding = 5
        bg_bbox = [
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        ]
        draw.rectangle(bg_bbox, fill="black", outline="white", width=1)
        draw.text((10, y_offset), text_line, fill="white", font=font)
        y_offset = text_bbox[3] + padding + 5
    
    # Display or save
    if output_path:
        draw_img.save(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        draw_img.show()
        print(f"Displayed image with {len(coords)} predicted coordinate(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predicted coordinates on screenshots"
    )
    
    # Two modes: from summary JSON or direct input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--summary_json",
        type=str,
        help="Path to uitars_summary.json file",
    )
    input_group.add_argument(
        "--image_path",
        type=str,
        help="Direct path to image file",
    )
    
    # Arguments for summary JSON mode
    parser.add_argument(
        "--episode_id",
        type=str,
        help="Episode ID (required if --summary_json is provided)",
    )
    parser.add_argument(
        "--step_index",
        type=int,
        default=None,
        help="Step index to visualize (if not provided, visualizes all steps)",
    )
    
    # Arguments for direct mode
    parser.add_argument(
        "--prediction",
        type=str,
        help="Prediction text (required if --image_path is provided)",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        nargs="*",
        help="Ground truth actions (optional)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction text (optional)",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (if not provided, displays interactively)",
    )
    
    args = parser.parse_args()
    
    if args.summary_json:
        # Load summary JSON
        with open(args.summary_json, "r") as f:
            summary = json.load(f)
        
        if not args.episode_id:
            parser.error("--episode_id is required when using --summary_json")
        
        # Find episode
        episode = None
        for ep in summary.get("episodes", []):
            if ep.get("episode") == args.episode_id:
                episode = ep
                break
        
        if not episode:
            print(f"Episode {args.episode_id} not found in summary")
            return
        
        # Get steps to visualize
        steps = episode.get("steps", [])
        if args.step_index is not None:
            steps = [s for s in steps if s.get("step_index") == args.step_index]
            if not steps:
                print(f"Step {args.step_index} not found in episode")
                return
        
        # Visualize each step
        for step in steps:
            screenshot_path = step.get("screenshot_path")
            if not screenshot_path:
                print(f"Warning: No screenshot_path for step {step.get('step_index')}")
                continue
            
            # Check if path is absolute or relative
            if not Path(screenshot_path).is_absolute():
                # Try relative to summary JSON directory
                summary_dir = Path(args.summary_json).parent
                screenshot_path = summary_dir / screenshot_path
                if not screenshot_path.exists():
                    # Try as-is
                    screenshot_path = step.get("screenshot_path")
            
            prediction = step.get("prediction", "")
            ground_truth = step.get("ground_truth", [])
            instruction = step.get("instruction", "")
            step_idx = step.get("step_index", "unknown")
            
            output_path = None
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"episode_{args.episode_id}_step_{step_idx}.png")
            
            print(f"\nVisualizing step {step_idx}...")
            visualize_prediction(
                image_path=str(screenshot_path),
                prediction_text=prediction,
                ground_truth=ground_truth,
                instruction=instruction,
                output_path=output_path,
            )
    
    else:
        # Direct mode
        if not args.prediction:
            parser.error("--prediction is required when using --image_path")
        
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = Path(args.image_path).stem
            output_path = str(output_dir / f"{image_name}_prediction.png")
        
        visualize_prediction(
            image_path=args.image_path,
            prediction_text=args.prediction,
            ground_truth=args.ground_truth,
            instruction=args.instruction,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()