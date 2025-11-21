#!/usr/bin/env python3
"""
Script for UITARS15_v1, based on https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/uitars15_v1.py

Designed to work with vLLM serving models like:
    vllm serve "ByteDance-Seed/UI-TARS-1.5-7B"

Usage:
    python standalone_predict.py image.png "Click on the login button"    


Raw Prediction Example:
```
Action: click(start_box='(1479,503)')
```

save:
1. predictions

metrics:
0. ~~action str exact match~~
1. hit box accuracy
2. MSE(distance to center of the bounding box)
"""

import sys
import os
import ast
import base64
import math
import re
import argparse
import json
from collections import OrderedDict
from collections import deque
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Sequence, Any, Deque
from PIL import Image

from eval.mind2web_mapping import uitars_action_to_mind2web_op

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

# ============================================================================
# Constants and Prompts (from uitars15_v1.py)
# ============================================================================

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
{action_space}
## User Instruction
{instruction}
"""

# ============================================================================
# Helper Functions (from uitars15_v1.py)
# ============================================================================

def parse_action(action_str):
    """Parse an action string into function name and arguments."""
    try:
        node = ast.parse(action_str, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")
        
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None
        
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value
        
        return {'function': func_name, 'args': kwargs}
    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    """Escape single quotes in text."""
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """Rescale image dimensions to meet constraints."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}")
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

def pil_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, 
                                     model_type, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS):
    """Parse model output text into structured actions."""
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height, origin_resized_width, 
            factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
        )
    
    # Extract thought
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    assert "Action:" in text, "No Action found in response"
    action_str = text.split("Action:")[-1]
    
    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            pattern = r"type\(content='(.*?)'\)"
            content = re.sub(pattern, lambda m: m.group(1), action_str)
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)
    
    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            raise ValueError(f"Action can't parse: {raw_str}")
        
        action_type = action_instance["function"]
        params = action_instance["args"]
        
        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                numbers = ori_box.replace("(", "").replace(")", "").split(",")
                
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]
                
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], 
                                   float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)
        
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


# ============================================================================
# Evaluation Helpers and Agent Wrapper
# ============================================================================

def _split_action_strings(prediction_text: str) -> List[str]:
    """
    Extract raw action strings from a model response while preserving formatting.
    """
    if not prediction_text or "Action:" not in prediction_text:
        return []

    tail = prediction_text.split("Action:", 1)[1]
    lines = tail.replace("\r", "").splitlines()
    terminators = ("Thought:", "Reflection:", "Summary:", "Observation:", "Call_user", "Call User")
    buffer: List[str] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            joined = " ".join(segment.strip() for segment in current if segment.strip())
            if joined:
                buffer.append(joined)
        current = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue
        if any(stripped.startswith(term) for term in terminators):
            flush()
            break
        current.append(stripped)
    flush()
    return buffer


def _parse_start_point(action_inputs: Dict[str, str], img_w: Optional[int], img_h: Optional[int]) -> Optional[Tuple[float, float]]:
    """
    Convert the first available start_box/end_box entry into absolute pixel coordinates.
    """
    candidate = None
    for key in ("start_box", "end_box"):
        if key in action_inputs:
            candidate = action_inputs[key]
            break
    if candidate is None:
        return None
    try:
        if isinstance(candidate, str):
            coords = ast.literal_eval(candidate)
        else:
            coords = candidate
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return None
        x_raw = float(coords[0])
        y_raw = float(coords[1])
        if img_w and 0.0 <= x_raw <= 1.0:
            x = x_raw * img_w
        else:
            x = x_raw
        if img_h and 0.0 <= y_raw <= 1.0:
            y = y_raw * img_h
        else:
            y = y_raw
        return x, y
    except Exception:
        return None


def _point_inside_bbox(point: Tuple[float, float], bbox: Sequence[float]) -> bool:
    x, y = point
    if len(bbox) < 4:
        return False
    bx, by, bw, bh = map(float, bbox[:4])
    return bx <= x <= bx + bw and by <= y <= by + bh


def _center_from_bbox(bbox: Sequence[float]) -> Optional[Tuple[float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x, y, w, h = map(float, bbox[:4])
        return x + w / 2.0, y + h / 2.0
    except (TypeError, ValueError):
        return None


def _mse_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy)


def _normalize_action_sequence(actions: Sequence[str]) -> List[str]:
    return [action.strip() for action in actions if action and action.strip()]


class UITARSAgent:
    """
    Lightweight UITARS agent wrapper that adapts episode loader output to the vLLM-style API.

    This class focuses on:
      - Building the prompt using UITARS templates from this module.
      - Converting screenshot bytes to base64-encoded images.
      - Maintaining a simple sliding window of past screenshots (history_n).
      - Calling an OpenAI-compatible client and returning (prediction_text, actions).

    It intentionally reuses the helper functions and prompt constants defined above,
    rather than re-implementing the original OSWorld UITARSAgent in full.
    """

    def __init__(
        self,
        model: str,
        runtime_conf: Dict[str, Any],
        observation_type: str = "screenshot",
        model_type: str = "qwen25vl",
    ) -> None:
        self.model = model
        self.runtime_conf = dict(runtime_conf)
        self.observation_type = observation_type
        self.model_type = model_type

        # Core generation/config parameters
        self.temperature: float = float(self.runtime_conf.get("temperature", 0.0))
        self.top_p: float = float(self.runtime_conf.get("top_p", 0.9))
        self.max_tokens: int = int(self.runtime_conf.get("max_tokens", 512))
        self.language: str = str(self.runtime_conf.get("language", "English"))
        self.seed: Optional[int] = self.runtime_conf.get("seed")

        # Image constraints
        self.max_pixels: int = int(self.runtime_conf.get("max_pixels", MAX_PIXELS))
        self.min_pixels: int = int(self.runtime_conf.get("min_pixels", MIN_PIXELS))

        # History control: how many past screenshots to send, including current.
        self.history_n: int = int(self.runtime_conf.get("history_n", 5))
        if self.history_n < 1:
            self.history_n = 1

        # Sliding window of screenshot bytes (oldest first).
        self._screenshot_history: Deque[bytes] = deque(maxlen=self.history_n)

    def reset(self) -> None:
        """Clear internal history so the next step is stateless."""
        self._screenshot_history.clear()

    def _build_messages(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Build OpenAI-compatible messages with the current history of screenshots.

        The user message contains:
          - A single text entry with the UITARS prompt (including action space).
          - One image entry per screenshot in the history, oldest to newest.
        """
        # Format textual prompt with action space and instruction.
        prompt = UITARS_USR_PROMPT_THOUGHT.format(
            action_space=UITARS_ACTION_SPACE,
            language=self.language,
            instruction=instruction,
        )

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Attach each screenshot in history as an image_url.
        for screenshot_bytes in self._screenshot_history:
            try:
                image = Image.open(BytesIO(screenshot_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception:
                # If a particular frame cannot be decoded, skip it.
                continue

            encoded = pil_to_base64(image)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Run one prediction step for a given instruction and observation.

        Args:
            instruction: Text instruction for this step.
            obs: Observation dict from episode_loader, expected to contain:
                - "screenshot": bytes
                - "accessibility_tree": currently unused

        Returns:
            prediction_text: Raw model text response.
            actions: List of raw UITARS action strings parsed from the response.
        """
        screenshot_bytes = obs.get("screenshot")
        if not isinstance(screenshot_bytes, (bytes, bytearray)):
            raise ValueError("obs['screenshot'] must be bytes.")

        # Update sliding window with the current frame.
        self._screenshot_history.append(bytes(screenshot_bytes))

        messages = self._build_messages(instruction)

        # Resolve API endpoint and key, preferring DOUBAO_* but falling back to VLLM_*.
        api_url = os.environ.get("DOUBAO_API_URL") or os.environ.get(
            "VLLM_API_URL", "http://localhost:8000/v1"
        )
        api_key = os.environ.get("DOUBAO_API_KEY") or os.environ.get(
            "VLLM_API_KEY", "EMPTY"
        )

        client = OpenAI(base_url=api_url, api_key=api_key)

        # Build request kwargs, including seed if provided
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            request_kwargs["seed"] = self.seed

        response = client.chat.completions.create(**request_kwargs)
        prediction_text = response.choices[0].message.content.strip()

        # Extract raw UITARS action strings from the response.
        actions = _split_action_strings(prediction_text)
        return prediction_text, actions


def compute_step_metrics(
    prediction_text: str,
    screenshot_bytes: bytes,
    metadata: Dict,
    model_type: str = "qwen25vl",
    max_pixels: int = MAX_PIXELS,
    min_pixels: int = MIN_PIXELS,
) -> Dict[str, Optional[float]]:
    """
    Compute per-step evaluation metrics.

    Returns:
        {
            "action_str_em": Optional[float],
            "hit_box_accuracy": Optional[float],
            "bbox_center_mse": Optional[float],
        }
    """
    metrics: Dict[str, Optional[float]] = OrderedDict(
        (
            ("action_str_em", None),
            ("hit_box_accuracy", None),
            ("bbox_center_mse", None),
        )
    )

    image_w = image_h = None
    parsed_actions: List[Dict] = []
    predicted_point: Optional[Tuple[float, float]] = None

    if screenshot_bytes:
        try:
            screenshot = Image.open(BytesIO(screenshot_bytes))
            image_w, image_h = screenshot.size
        except Exception:
            pass

    if prediction_text and image_w and image_h:
        try:
            parsed_actions = parse_action_to_structure_output(
                prediction_text,
                factor=1000,
                origin_resized_height=image_h,
                origin_resized_width=image_w,
                model_type=model_type,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
        except Exception:
            parsed_actions = []

    ground_truth_op = metadata.get("op")
    if ground_truth_op and parsed_actions:
        first_action = parsed_actions[0]
        parsed_action_type = first_action.get("action_type")
        parsed_inputs = first_action.get("action_inputs", {})
        
        # Convert UITARS action type to Mind2Web op
        predicted_op = uitars_action_to_mind2web_op(parsed_action_type, parsed_inputs)
        
        if predicted_op is not None:
            gt_op_normalized = ground_truth_op.upper().strip()
            pred_op_normalized = predicted_op.upper().strip()
            metrics["action_str_em"] = 1.0 if gt_op_normalized == pred_op_normalized else 0.0

    for parsed in parsed_actions:
        candidate = _parse_start_point(parsed.get("action_inputs", {}), image_w, image_h)
        if candidate:
            predicted_point = candidate
            break

    bbox = metadata.get("bounding_box")
    coords = metadata.get("coordinates") or []
    target_point = metadata.get("target_point") or _center_from_bbox(bbox)
    if target_point is None and len(coords) >= 2:
        try:
            target_point = (float(coords[0]), float(coords[1]))
        except (TypeError, ValueError):
            target_point = None

    if predicted_point is not None and bbox:
        metrics["hit_box_accuracy"] = (
            1.0 if _point_inside_bbox(predicted_point, bbox) else 0.0
        )

    if predicted_point is not None and target_point is not None:
        metrics["bbox_center_mse"] = _mse_distance(predicted_point, target_point)

    return metrics

# ============================================================================
# Main Prediction Function
# ============================================================================

def predict_action(image_path: str, instruction: str, 
                  model: str = "ByteDance-Seed/UI-TARS-1.5-7B",
                  api_url: str = None,
                  api_key: str = None,
                  temperature: float = 0.7,
                  max_tokens: int = 2048,
                  model_type: str = "qwen25vl",
                  language: str = "English",
                  max_pixels: int = MAX_PIXELS,
                  min_pixels: int = MIN_PIXELS,
                  output_json: bool = False) -> Dict:
    """
    Predict action from image and instruction.
    
    Args:
        image_path: Path to image file
        instruction: Text instruction
        model: Model name (defaults to ByteDance-Seed/UI-TARS-1.5-7B)
        api_url: API base URL (defaults to vLLM at http://localhost:8000/v1)
        api_key: API key (defaults to "EMPTY" for vLLM)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        model_type: "qwen25vl" or "qwen2vl"
        language: Language for thought output
        max_pixels: Maximum image pixels
        min_pixels: Minimum image pixels
        output_json: Output as JSON
    
    Returns:
        Dictionary with prediction results
    """
    # Load and process image
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Resize image if needed
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width = int(image.width * resize_factor)
        height = int(image.height * resize_factor)
        image = image.resize((width, height))
    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width = math.ceil(image.width * resize_factor)
        height = math.ceil(image.height * resize_factor)
        image = image.resize((width, height))
    
    origin_resized_height = image.height
    origin_resized_width = image.width
    
    # Encode image
    encoded_string = pil_to_base64(image)
    
    # Format prompt
    prompt = UITARS_USR_PROMPT_THOUGHT.format(
        action_space=UITARS_ACTION_SPACE,
        language=language,
        instruction=instruction
    )
    
    # Create messages in vLLM format (OpenAI-compatible)
    # Format matches official vLLM API:
    # {
    #   "role": "user",
    #   "content": [
    #     {"type": "text", "text": "..."},
    #     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    #   ]
    # }
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
            ]
        }
    ]
    
    # Setup API client for vLLM server
    # vLLM serves models via OpenAI-compatible REST API at http://localhost:8000/v1/chat/completions
    # Official command: vllm serve "ByteDance-Seed/UI-TARS-1.5-7B"
    if api_url is None:
        api_url = os.environ.get('VLLM_API_URL', 'http://localhost:8000/v1')
    if api_key is None:
        api_key = os.environ.get('VLLM_API_KEY', 'EMPTY')
    
    # Create OpenAI client pointing to vLLM server
    # The client automatically appends /chat/completions to the base URL
    client = OpenAI(base_url=api_url, api_key=api_key)
    
    # Call vLLM server via OpenAI-compatible API
    # This matches the official curl format:
    # curl -X POST "http://localhost:8000/v1/chat/completions" \
    #   -H "Content-Type: application/json" \
    #   --data '{"model": "ByteDance-Seed/UI-TARS-1.5-7B", "messages": [...]}'
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"vLLM API call failed: {e}\n"
            f"Make sure vLLM server is running: vllm serve 'ByteDance-Seed/UI-TARS-1.5-7B'\n"
            f"API endpoint: {api_url}/chat/completions"
        )
    
    # Parse prediction
    try:
        parsed_actions = parse_action_to_structure_output(
            prediction,
            factor=1000,
            origin_resized_height=origin_resized_height,
            origin_resized_width=origin_resized_width,
            model_type=model_type,
            max_pixels=max_pixels,
            min_pixels=min_pixels
        )
    except Exception as e:
        raise ValueError(f"Failed to parse prediction: {e}\nRaw prediction: {prediction}")
    
    # Format output
    result = {
        "prediction": prediction,
        "image_size": {"width": origin_resized_width, "height": origin_resized_height},
        "actions": []
    }
    
    for parsed_action in parsed_actions:
        action_data = {
            "action_type": parsed_action.get("action_type"),
            "action_inputs": parsed_action.get("action_inputs", {}),
            "thought": parsed_action.get("thought"),
            "reflection": parsed_action.get("reflection")
        }
        result["actions"].append(action_data)
    
    return result

# ============================================================================
# CLI Interface
# ============================================================================

def format_coordinates(action_inputs: Dict) -> str:
    """Format coordinates for display."""
    coords = []
    if "start_box" in action_inputs:
        start_box = action_inputs["start_box"]
        try:
            coords_list = eval(start_box) if isinstance(start_box, str) else start_box
            if len(coords_list) >= 2:
                coords.append(f"Start: ({coords_list[0]:.4f}, {coords_list[1]:.4f})")
            if len(coords_list) >= 4:
                coords.append(f"End: ({coords_list[2]:.4f}, {coords_list[3]:.4f})")
        except:
            coords.append(f"Start Box: {start_box}")
    
    if "end_box" in action_inputs:
        end_box = action_inputs["end_box"]
        try:
            coords_list = eval(end_box) if isinstance(end_box, str) else end_box
            if len(coords_list) >= 2:
                coords.append(f"End: ({coords_list[0]:.4f}, {coords_list[1]:.4f})")
        except:
            coords.append(f"End Box: {end_box}")
    
    return ", ".join(coords) if coords else "No coordinates"

def main():
    parser = argparse.ArgumentParser(
        description="Standalone GUI action prediction from image and text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_predict.py image.png "Click on the login button"
  python standalone_predict.py image.png "Type 'hello' in the search box" --model ByteDance-Seed/UI-TARS-1.5-7B
  python standalone_predict.py image.png "Click login" --api-url http://localhost:8000/v1 --output-json
        """
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("instruction", help="Text instruction")
    parser.add_argument("--model", default="ByteDance-Seed/UI-TARS-1.5-7B", 
                       help="Model name (default: ByteDance-Seed/UI-TARS-1.5-7B)")
    parser.add_argument("--api-url", help="API base URL (default: http://localhost:8000/v1 for vLLM)")
    parser.add_argument("--api-key", help="API key (default: 'EMPTY' for vLLM)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens (default: 2048)")
    parser.add_argument("--model-type", default="qwen25vl", choices=["qwen25vl", "qwen2vl"],
                       help="Model type (default: qwen25vl)")
    parser.add_argument("--language", default="English", help="Language for thought (default: English)")
    parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Make prediction
    try:
        result = predict_action(
            image_path=args.image,
            instruction=args.instruction,
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model_type=args.model_type,
            language=args.language,
            output_json=args.output_json
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Output results
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"\nImage Size: {result['image_size']['width']}x{result['image_size']['height']}")
        print(f"\nRaw Prediction:\n{result['prediction']}\n")
        print("-" * 80)
        print("Parsed Actions:")
        print("-" * 80)
        
        for i, action in enumerate(result["actions"], 1):
            print(f"\nAction {i}:")
            print(f"  Type: {action['action_type']}")
            
            if action.get("thought"):
                print(f"  Thought: {action['thought']}")
            if action.get("reflection"):
                print(f"  Reflection: {action['reflection']}")
            
            coords_str = format_coordinates(action.get("action_inputs", {}))
            if coords_str != "No coordinates":
                print(f"  Coordinates: {coords_str}")
            
            other_params = {k: v for k, v in action.get("action_inputs", {}).items() 
                          if k not in ["start_box", "end_box"]}
            if other_params:
                print(f"  Parameters: {other_params}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
