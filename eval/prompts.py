"""
Prompt templates for different model types and reasoning configurations.
"""

import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Tuple
import math

EXPECTED_IMAGE_WIDTH = 1920
EXPECTED_IMAGE_HEIGHT = 1080

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

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

def resize_image(image: Image.Image) -> Image.Image:
    """Resize image to expected resolution."""
    original_width, original_height = image.size
    
    # Apply smart resize
    resized_height, resized_width = smart_resize(
        original_height,
        original_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    
    # Resize the image
    resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    return resized_image

def convert_pil_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ============================================================================
# Prompt Templates
# ============================================================================

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
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

## Note
- Use English in `Thought` part.
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

GTA1_SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
"""

GTA1_SYSTEM_PROMPT_THOUGHT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

## Output Format
```
Thought: ...
Action: (x,y)
```

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

Then output the coordinate pair exactly:
(x,y)
"""


QWEN25_SYSTEM_PROMPT_NOTHOUGHT = (
    "You are a helpful assistant.\n\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{\"type\": \"function\", \"function\": {\"name_for_human\": \"computer_use\", \"name\": \"computer_use\", \"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is {screen_width}x{screen_height}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", \"right_click\", \"middle_click\", \"double_click\", \"scroll\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"keys\": {\"description\": \"Required only by `action=key`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.\", \"type\": \"array\"}, \"pixels\": {\"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.\", \"type\": \"number\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=wait`.\", \"type\": \"number\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}, \"args_format\": \"Format the arguments as a JSON object.\"}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"
)

QWEN25_SYSTEM_PROMPT_THOUGHT = (
    "You are a helpful assistant.\n\n\n"
    "# Output Format\n\n"
    "Before making a tool call, you should think through your approach. Use the following format:\n\n"
    "Thought: [Write a small plan analyzing the current screenshot, identifying the target element(s), and summarizing your next action with its target element in one sentence.]\n\n"
    "Then make your tool call.\n\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{\"type\": \"function\", \"function\": {\"name_for_human\": \"computer_use\", \"name\": \"computer_use\", \"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is {screen_width}x{screen_height}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", \"right_click\", \"middle_click\", \"double_click\", \"scroll\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"keys\": {\"description\": \"Required only by `action=key`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.\", \"type\": \"array\"}, \"pixels\": {\"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.\", \"type\": \"number\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=wait`.\", \"type\": \"number\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}, \"args_format\": \"Format the arguments as a JSON object.\"}\n"
    "</tools>\n\n"
    "For each function call, first write your Thought, then return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "Thought: [Your reasoning and plan]\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"
)


# ============================================================================
# Build Messages
# ============================================================================

def build_gta1_messages(instruction: str, image: Image.Image, use_reasoning: bool) -> List[Dict[str, Any]]:
    resized_image = resize_image(image)
    encoded_resized_image = convert_pil_image_to_base64(resized_image)
    
    # Select appropriate prompt template
    if use_reasoning:
        system_prompt = GTA1_SYSTEM_PROMPT_THOUGHT.format(height=image.height, width=image.width)
    else:
        system_prompt = GTA1_SYSTEM_PROMPT.format(height=image.height, width=image.width)
    
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_resized_image}"}},
            {"type": "text", "text": instruction}
        ]
    }

    return [system_message, user_message]


def build_uitars15_messages(instruction: str, image: Image.Image, use_reasoning: bool) -> List[Dict[str, Any]]:
    resized_image = resize_image(image)
    encoded_resized_image = convert_pil_image_to_base64(resized_image)

    if use_reasoning:
        prompt = UITARS_USR_PROMPT_THOUGHT.format(instruction=instruction)
    else:
        prompt = UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=instruction)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_resized_image}"}}]
        }
    ]
    return messages


def build_qwen25vl_messages(instruction: str, image: Image.Image, use_reasoning: bool) -> List[Dict[str, Any]]:
    """
    Build messages for Qwen2.5VL following the official example format.
    
    Args:
        instruction: Text instruction for the model
        image: PIL Image (should be resized)
        screen_width: Width of the resized image
        screen_height: Height of the resized image
        use_reasoning: Whether to use reasoning prompt template
    
    Returns:
        List of message dictionaries in OpenAI format
    """
    # Encode image to base64
    resized_image = resize_image(image)
    encoded_resized_image = convert_pil_image_to_base64(resized_image)
    
    # Select appropriate prompt template
    if use_reasoning:
        prompt_template = QWEN25_SYSTEM_PROMPT_THOUGHT.replace("{screen_width}", str(resized_image.width)).replace("{screen_height}", str(resized_image.height))
    else:
        prompt_template = QWEN25_SYSTEM_PROMPT_NOTHOUGHT.replace("{screen_width}", str(resized_image.width)).replace("{screen_height}", str(resized_image.height))
    
    # Split system prompt into first line and rest
    # The first line is "You are a helpful assistant." or similar
    lines = prompt_template.split('\n', 1)
    first_line = lines[0] if lines else "You are a helpful assistant."
    rest_of_prompt = lines[1] if len(lines) > 1 else ""
    
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": first_line
                },
                {
                    "type": "text",
                    "text": rest_of_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + encoded_resized_image
                    }
                },
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
    ]
