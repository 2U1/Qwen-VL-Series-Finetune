"""
Utilities to translate Mind2Web-style recorded actions into UITARS action strings.
"""
from typing import Dict, List, Optional, Tuple, Union


def _escape_single_quotes(text: str) -> str:
    """Escape single quotes for safe embedding inside single-quoted UITARS strings."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _center_of_bbox(bbox: List[Union[int, float]]) -> Optional[Tuple[int, int]]:
    """
    Compute the center point (x, y) of a bounding box defined as [x, y, w, h].
    Returns None if bbox is invalid.
    """
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        return cx, cy
    except Exception:
        return None


def _first_xy_from_coordinates(coords: List[Union[int, float]]) -> Optional[Tuple[int, int]]:
    """
    Extract the first (x, y) pair from the Mind2Web 'coordinates' field.
    """
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        x, y = int(round(float(coords[0]))), int(round(float(coords[1])))
        return x, y
    except Exception:
        return None


def _format_box(x: int, y: int) -> str:
    """Format a coordinate pair into UITARS box token."""
    return f"<|box_start|>({x},{y})<|box_end|>"


def _maybe_click_action(step: Dict) -> Optional[str]:
    """Build a UITARS click action if we can resolve a target point."""
    coords = step.get("coordinates") or []
    bbox = step.get("bounding_box") or []
    point = _first_xy_from_coordinates(coords) or _center_of_bbox(bbox)
    if point is None:
        return None
    bx = _format_box(*point)
    return f"click(start_box='{bx}')"


def _type_action_from_value(value: Optional[str]) -> Optional[str]:
    """Build a UITARS type action from a given content string."""
    if not value:
        return None
    content = _escape_single_quotes(str(value))
    return f"type(content='{content}')"


def _scroll_action(step: Dict) -> Optional[str]:
    """
    Build a UITARS scroll action.
    Expects optional keys:
      - coordinates: [x, y]
      - bounding_box: [x, y, w, h] (fallback for center)
      - scroll_direction: 'down'|'up'|'left'|'right' (default 'down')
    """
    direction = step.get("scroll_direction", "down")
    if direction not in ("down", "up", "left", "right"):
        direction = "down"
    coords = step.get("coordinates") or []
    bbox = step.get("bounding_box") or []
    point = _first_xy_from_coordinates(coords) or _center_of_bbox(bbox)
    if point is None:
        return None
    bx = _format_box(*point)
    return f"scroll(start_box='{bx}', direction='{direction}')"


def _hotkey_action(step: Dict) -> Optional[str]:
    """
    Build UITARS hotkey action from Mind2Web-like fields.
    Expects key value via any of:
      - step['type_action_value']
      - step['key']
      - step['hotkey']
    """
    key = step.get("type_action_value") or step.get("key") or step.get("hotkey")
    if not key:
        return None
    key_str = _escape_single_quotes(str(key))
    return f"hotkey(key='{key_str}')"


def mind2web_step_to_uitars(step: Dict) -> List[str]:
    """
    Translate a single Mind2Web step dict to one or more UITARS action strings.
    Returns an empty list when a mapping cannot be determined.
    """
    op = (step.get("op") or "").upper()
    actions: List[str] = []

    if op == "CLICK":
        click = _maybe_click_action(step)
        if click:
            actions.append(click)

    elif op == "HOVER":
        # UITARS has no explicit hover; approximate as a click on the target.
        click = _maybe_click_action(step)
        if click:
            actions.append(click)

    elif op == "TYPE":
        # Prefer to click into the target if we can locate it, then type.
        click = _maybe_click_action(step)
        if click:
            actions.append(click)
        type_val = _type_action_from_value(step.get("type_action_value"))
        if type_val:
            actions.append(type_val)

    elif op == "SELECT":
        # Many M2W 'SELECT' steps correspond to choosing a value in a combobox.
        # Heuristic: click the control then type the desired value (if provided).
        click = _maybe_click_action(step)
        if click:
            actions.append(click)
        type_val = _type_action_from_value(step.get("type_action_value"))
        if type_val:
            actions.append(type_val)

    elif op in ("PRESS ENTER", "PRESS_ENTER", "ENTER", "RETURN"):
        # Submit via newline per UITARS guidance
        actions.append("type(content='\\n')")

    elif op == "SCROLL":
        scroll = _scroll_action(step)
        if scroll:
            actions.append(scroll)

    elif op in ("KEYPRESS", "KEY_PRESS", "HOTKEY"):
        hotkey = _hotkey_action(step)
        if hotkey:
            actions.append(hotkey)

    elif op == "DRAG":
        # Optional mapping if Mind2Web provides drag coordinates:
        # expected fields could be drag_start=[x1,y1], drag_end=[x2,y2]
        start = _first_xy_from_coordinates(step.get("drag_start") or [])
        end = _first_xy_from_coordinates(step.get("drag_end") or [])
        if start and end:
            start_box = _format_box(*start)
            end_box = _format_box(*end)
            actions.append(f"drag(start_box='{start_box}', end_box='{end_box}')")

    elif op == "IGNORE":
        actions.append("wait()")

    # Other ops could be added here (e.g., RIGHT_CLICK, DOUBLE_CLICK) if present in data.
    return actions


def uitars_action_to_mind2web_op(action_type: str, action_inputs: Dict[str, str]) -> Optional[str]:
    """
    Map UITARS action_type to Mind2Web 'op' string.
    
    Args:
        action_type: UITARS action name (e.g., 'click', 'type', 'scroll')
        action_inputs: Dictionary of action parameters (e.g., {'content': 'hello'})
    
    Returns:
        Mind2Web operation code (e.g., 'CLICK', 'TYPE', 'SCROLL')
    
    Examples:
        >>> uitars_action_to_mind2web_op('click', {})
        'CLICK'
        >>> uitars_action_to_mind2web_op('type', {'content': '\\n'})
        'PRESS ENTER'
        >>> uitars_action_to_mind2web_op('hotkey', {'key': 'enter'})
        'HOTKEY'
    """
    action_type = (action_type or "").lower().strip()
    
    # Direct mappings
    ACTION_TYPE_TO_OP = {
        "click": "CLICK",
        "left_double": "CLICK",  # Mind2Web doesn't distinguish double-click, map to CLICK
        "right_single": "CLICK",  # Mind2Web doesn't have right-click, map to CLICK
        "drag": "DRAG",
        "scroll": "SCROLL",
        "wait": "IGNORE",
        "finished": "IGNORE",  # No direct Mind2Web equivalent
    }
    
    # Check for direct mapping
    if action_type in ACTION_TYPE_TO_OP:
        return ACTION_TYPE_TO_OP[action_type]
    
    # Special handling for type action
    if action_type == "type":
        content = action_inputs.get("content", "")
        # Check if it's a submit action (ends with newline)
        if content and content.strip() == "\\n":
            return "ENTER"
        return "TYPE"
    
    # Special handling for hotkey
    if action_type == "hotkey":
        key = action_inputs.get("key", "").lower()
        # Map common hotkeys to specific Mind2Web ops
        if key in ("enter", "return", "\\n"):
            return "ENTER"
        return "HOTKEY"
    
    return None

