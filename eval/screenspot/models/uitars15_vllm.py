"""
UI-TARS1.5 model wrapper for vLLM API inference in ScreenSpot evaluation.
"""

import os
import re
import ast
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image
import sys

# Add eval directory to path for imports
eval_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(eval_dir))

from vllm_client import ModelClient, ModelConfig
# Import smart_resize from prompts to ensure consistency
from prompts import smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS


def escape_single_quotes(text):
    """Match unescaped single quotes (not matching \\')."""
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def parse_action(action_str):
    """Parse action string using AST."""
    try:
        # Parse string as AST node
        node = ast.parse(action_str, mode='eval')

        # Ensure node is an expression
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # Get expression body
        call = node.body

        # Ensure body is a function call
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # Get function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # Get keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # Handle different value types, assume constants
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # Compatibility with older Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        # Fail immediately - let debugger catch this
        raise ValueError(f"Failed to parse action string '{action_str}': {e}") from e


# Global counter for tracking parsing failures
_parse_failures = {'count': 0, 'details': []}


def parse_action_to_structure_output(
    text, 
    factor=IMAGE_FACTOR, 
    origin_resized_height=1080, 
    origin_resized_width=1920, 
    model_type="uitars15", 
    max_pixels=16384*28*28, 
    min_pixels=100*28*28
):
    """
    Parse UI-TARS1.5 action text to structured output.
    
    Args:
        text: Raw model response text
        factor: Image resize factor (default: IMAGE_FACTOR)
        origin_resized_height: ACTUAL original image height (NOT hardcoded)
        origin_resized_width: ACTUAL original image width (NOT hardcoded)
        model_type: Model type identifier
        max_pixels: Maximum pixels for smart_resize
        min_pixels: Minimum pixels for smart_resize
    
    Returns:
        List of structured action dictionaries
        
    Raises:
        ValueError: If parsing fails or unexpected format encountered
        AssertionError: If required "Action:" not found in text
    """
    if origin_resized_height <= 0 or origin_resized_width <= 0:
        raise ValueError(
            f"Invalid image dimensions: height={origin_resized_height}, width={origin_resized_width}. "
            f"Both must be positive integers."
        )
    
    text = text.strip()
    if not text:
        raise ValueError("Empty text provided to parse_action_to_structure_output")
    
    if model_type == "uitars15":
        # Calculate smart_resize dimensions using ACTUAL image dimensions
        # This must match exactly what build_uitars15_messages does in prompts.py
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height, 
            origin_resized_width, 
            factor=IMAGE_FACTOR, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'uitars15'.")

    # Regex to match Action string
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    if "Action:" not in text:
        raise AssertionError(
            f"Required 'Action:' not found in model response. "
            f"Text (first 500 chars): {text[:500]}"
        )
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # Regex to match content string and escape single quotes
            def escape_quotes(match):
                content = match.group(1)  # Get content value
                return content

            # Use regex for replacement
            pattern = r"type\(content='(.*?)'\)"  # Match type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # Process string
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            # Fail immediately - let debugger catch this
            raise ValueError(
                f"Failed to parse action string. "
                f"Raw action string: {raw_str}. "
                f"Full text (first 1000 chars): {text[:1000]}"
            )
        
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()  # Remove quotes and extra spaces
            # Handle start_box or end_box parameter format
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and brackets, then split by commas
                numbers = ori_box.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
                # Convert to float and scale
                # UI-TARS outputs coordinates in resized image space
                if model_type == "uitars15":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        try:
                            # Clean number string: strip whitespace and remove trailing colons
                            num_cleaned = num.strip().rstrip(':').strip()
                            num = float(num_cleaned)
                            # Convert from resized space to original image space
                            # Use actual original dimensions (origin_resized_height/width) instead of hardcoded values
                            if (num_idx + 1) % 2 == 0:  # y coordinate (even index, 0-based)
                                float_numbers.append(round(float(num / smart_resize_height * origin_resized_height)))
                            else:  # x coordinate (odd index, 0-based)
                                float_numbers.append(round(float(num / smart_resize_width * origin_resized_width)))
                        except ValueError as e:
                            _parse_failures['count'] += 1
                            _parse_failures['details'].append({
                                'param_name': param_name,
                                'ori_box': ori_box,
                                'numbers': numbers,
                                'problematic_num': num,
                                'problematic_num_cleaned': num_cleaned,
                                'text_preview': text[:500]
                            })
                            # Fail immediately - let debugger catch this
                            raise ValueError(
                                f"Failed to parse number in box coordinates. "
                                f"Failure #{_parse_failures['count']}. "
                                f"Parameter name: {param_name}. "
                                f"Original box value: {ori_box}. "
                                f"Numbers list: {numbers}. "
                                f"Problematic number (index {num_idx}): original='{num}', cleaned='{num_cleaned}'. "
                                f"Raw prediction text (first 1000 chars): {text[:1000]}"
                            )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


def get_action_type_and_coordinates_from_structured_actions_for_uitars15(structured_actions):
    """
    Extract action type and coordinates from structured actions.
    
    Handles UI-TARS1.5's pyautogui action space: click, left_double, right_single.
    For ScreenSpot evaluation, all click-like actions are treated the same.
    
    Returns:
        (action_type, coordinates) where coordinates is [x, y] or None
    """
    # Handle None (parsing failures) - fail loud
    if structured_actions is None:
        raise ValueError("structured_actions is None - parsing failed")
    if len(structured_actions) == 0:
        raise ValueError("structured_actions is empty - no actions found")

    action_type = structured_actions[0]['action_type']
    
    # Handle all click-like actions (UI-TARS1.5 pyautogui action space)
    # For ScreenSpot, we treat click, left_double, and right_single the same way
    if action_type in ['click', 'left_double', 'right_single']:
        if 'start_box' not in structured_actions[0]['action_inputs']:
            raise ValueError(
                f"Action type '{action_type}' missing 'start_box' in action_inputs. "
                f"Available keys: {list(structured_actions[0]['action_inputs'].keys())}"
            )
        coordinates = structured_actions[0]['action_inputs']['start_box']
        coordinates = ast.literal_eval(coordinates)
        # Extract first two coordinates (x, y) if it's a box
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            return action_type, [coordinates[0], coordinates[1]]
        elif isinstance(coordinates, (list, tuple)) and len(coordinates) < 2:
            raise ValueError(
                f"Coordinates list/tuple has insufficient elements: {coordinates}. "
                f"Expected at least 2 elements [x, y]."
            )
        else:
            raise ValueError(
                f"Unexpected coordinates format: {coordinates} (type: {type(coordinates)}). "
                f"Expected list or tuple with at least 2 elements."
            )
    else:
        # For non-click actions, return None coordinates
        return action_type, None


def detect_negative_case_from_structured_actions(structured_actions) -> str:
    """
    Detect if prediction indicates negative case (element not found) from structured actions.
    
    Returns:
        "positive" if click action found
        "negative" if call_user(), finished(), wait(), or no action found
        "wrong_format" if parsing fails
    """
    if structured_actions is None or len(structured_actions) == 0:
        return "wrong_format"
    
    action_type = structured_actions[0]['action_type']
    
    # Check for click actions (positive case)
    if action_type in ['click', 'left_double', 'right_single']:
        return "positive"
    
    # Check for negative indicators
    if action_type in ['call_user', 'finished', 'wait']:
        return "negative"
    
    # Other actions (drag, scroll, type, hotkey) - treat as negative for negative case detection
    return "negative"


class UITARS15VLLMModel:
    """UI-TARS1.5 model wrapper for vLLM API inference."""
    
    def __init__(self):
        self.model_client: Optional[ModelClient] = None
        self.model_config: Optional[ModelConfig] = None
        self.api_url: Optional[str] = None
        self.api_key: Optional[str] = None
    
    def load_model(
        self,
        model_name_or_path: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize the model client.
        
        Args:
            model_name_or_path: Model name for vLLM (default: ByteDance-Seed/UI-TARS-1.5-7B)
            api_url: vLLM API URL (default: from VLLM_API_URL env or http://localhost:8000/v1)
            api_key: API key (default: from VLLM_API_KEY env or "EMPTY")
            use_reasoning: Whether to use reasoning prompt template
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        # Set defaults
        model_name = model_name_or_path or "ByteDance-Seed/UI-TARS-1.5-7B"
        self.api_url = api_url or os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        
        # Create model config
        self.model_config = ModelConfig(
            name=model_name,
            model_type="uitars15",
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Create model client
        self.model_client = ModelClient(self.model_config, self.api_url, self.api_key)
    
    def set_generation_config(self, temperature: float = 0.0, max_new_tokens: int = 256):
        """
        Update generation configuration.
        
        Args:
            temperature: Generation temperature
            max_new_tokens: Maximum new tokens to generate
        """
        if self.model_config is not None:
            self.model_config.temperature = temperature
            self.model_config.max_tokens = max_new_tokens
    
    def ground_only_positive(self, instruction: str, image: str) -> Dict[str, Any]:
        """
        Ground instruction to a point (positive case only).
        
        Args:
            instruction: Text instruction
            image: Path to image file
        
        Returns:
            Dictionary with:
                - "point": [x, y] normalized coordinates [0,1] or None
                - "raw_response": Raw model response text
        """
        if self.model_client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_path = Path(image)
        
        # Load image to get original dimensions
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        original_width, original_height = img.size
        
        # Get raw prediction
        raw_response = self.model_client.predict(
            instruction=instruction,
            image_path=image_path,
            use_pattern_matching=False,  # Direct image loading for ScreenSpot
        )
        
        # Parse to structured actions - let exceptions propagate for debugging
        structured_actions = parse_action_to_structure_output(
            raw_response,
            origin_resized_height=original_height,
            origin_resized_width=original_width,
            model_type="uitars15"
        )
        
        if structured_actions is None:
            raise ValueError(
                f"parse_action_to_structure_output returned None. "
                f"Raw response (first 500 chars): {raw_response[:500]}"
            )
        
        # Extract coordinates - let exceptions propagate for debugging
        action_type, coordinates = get_action_type_and_coordinates_from_structured_actions_for_uitars15(structured_actions)
        
        # For ScreenSpot, we only accept click-like actions (click, left_double, right_single)
        if coordinates is None:
            raise ValueError(
                f"Action type '{action_type}' does not provide coordinates. "
                f"Expected one of: ['click', 'left_double', 'right_single']. "
                f"Structured actions: {structured_actions}. "
                f"Raw response (first 500 chars): {raw_response[:500]}"
            )
        
        # Coordinates are already in original image space after parse_action_to_structure_output
        # Normalize to [0, 1] using actual image dimensions (not hardcoded values)
        x_norm = coordinates[0] / original_width if original_width > 0 else 0.0
        y_norm = coordinates[1] / original_height if original_height > 0 else 0.0
        
        # Clamp to [0, 1] range to handle any edge cases
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        return {
            "point": [x_norm, y_norm],
            "raw_response": raw_response
        }
    
    def ground_allow_negative(self, instruction: str, image: str) -> Dict[str, Any]:
        """
        Ground instruction, allowing negative case (element not found).
        
        Args:
            instruction: Text instruction
            image: Path to image file
        
        Returns:
            Dictionary with:
                - "result": "positive", "negative", or "wrong_format"
                - "raw_response": Raw model response text
        """
        if self.model_client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_path = Path(image)
        
        # Load image to get original dimensions (needed for parsing)
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        original_width, original_height = img.size
        
        # Get raw prediction
        raw_response = self.model_client.predict(
            instruction=instruction,
            image_path=image_path,
            use_pattern_matching=False,  # Direct image loading for ScreenSpot
        )
        
        # Parse to structured actions - let exceptions propagate for debugging
        structured_actions = parse_action_to_structure_output(
            raw_response,
            origin_resized_height=original_height,
            origin_resized_width=original_width,
            model_type="uitars15"
        )
        
        if structured_actions is None:
            raise ValueError(
                f"parse_action_to_structure_output returned None for negative case. "
                f"Raw response (first 500 chars): {raw_response[:500]}"
            )
        
        # Detect positive/negative case from structured actions
        result = detect_negative_case_from_structured_actions(structured_actions)
        
        return {
            "result": result,
            "raw_response": raw_response
        }
