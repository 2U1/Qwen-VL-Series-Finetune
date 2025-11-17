"""
Helpers for translating Mind2Web-style recorded actions into UITARS action strings.

The Mind2Web trajectories store user actions and coordinate information. This module
converts those raw step dictionaries into the string-based UITARS action format used
by the evaluation agent, **only** for the core Mind2Web action space:

    CLICK, TYPE, SELECT, HOVER, ENTER / PRESS ENTER
"""
from typing import Dict, List, Optional, Tuple, Union


def _escape_single_quotes(text: str) -> str:
    """
    Escape single quotes for safe embedding inside single-quoted UITARS strings.

    UITARS action strings are typically single-quoted, so any literal single quote
    in the payload must be escaped to avoid breaking the command.
    """
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _compute_bbox_center(bbox: List[Union[int, float]]) -> Optional[Tuple[int, int]]:
    """
    Compute the integer center point of a bounding box.

    Args:
        bbox: A list-like object representing [x, y, width, height] in pixels.

    Returns:
        A tuple of integer pixel coordinates (cx, cy) for the box center,
        or None if the bounding box is missing or malformed.
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


def _extract_first_xy_from_coordinates(
    coords: List[Union[int, float]]
) -> Optional[Tuple[int, int]]:
    """
    Extract the first (x, y) pair from the Mind2Web ``coordinates`` field.

    Args:
        coords: A flat list whose first two elements represent the x and y values.

    Returns:
        A tuple of integer pixel coordinates (x, y), or None if the list is
        missing or cannot be parsed as numeric coordinates.
    """
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        x, y = int(round(float(coords[0]))), int(round(float(coords[1])))
        return x, y
    except Exception:
        return None


def _format_uitars_box(x: int, y: int) -> str:
    """
    Format a coordinate pair into a UITARS box token.

    The UITARS convention wraps coordinates in ``<|box_start|>`` / ``<|box_end|>``
    markers so that the agent can reliably parse target locations.
    """
    return f"<|box_start|>({x},{y})<|box_end|>"


def _build_click_action(step: Dict) -> Optional[str]:
    """
    Build a UITARS ``click(...)`` action for a single Mind2Web step.

    The function resolves a best-effort target point using, in order of priority:
      1. The ``coordinates`` field, if provided.
      2. The center of the ``bounding_box`` field, if provided.

    Args:
        step: Raw Mind2Web step dictionary.

    Returns:
        A UITARS click action string, or None if no valid target point can be found.
    """
    coords = step.get("coordinates") or []
    bbox = step.get("bounding_box") or []
    point = _extract_first_xy_from_coordinates(coords) or _compute_bbox_center(bbox)
    if point is None:
        return None
    bx = _format_uitars_box(*point)
    return f"click(start_box='{bx}')"


def _build_type_action(value: Optional[str]) -> Optional[str]:
    """
    Build a UITARS ``type(...)`` action from a given content string.

    Args:
        value: Textual content to type into the current focus element.

    Returns:
        A UITARS type action string, or None if the input is empty or None.
    """
    if not value:
        return None
    content = _escape_single_quotes(str(value))
    return f"type(content='{content}')"


def mind2web_step_to_uitars(step: Dict) -> List[str]:
    """
    Translate a single Mind2Web step into one or more UITARS action strings.

    This is the main entry point used by the evaluation pipeline. It inspects
    the Mind2Web operation type (``op``) and delegates to the appropriate
    helper to construct concrete UITARS commands (click, type, select, hover, enter).

    Args:
        step: Raw Mind2Web step dictionary as loaded from a trajectory JSON file.

    Returns:
        A list of UITARS action strings. The list is empty when no mapping
        can be determined for the given step.
    """
    op = (step.get("op") or "").upper()
    actions: List[str] = []

    if op == "CLICK":
        click = _build_click_action(step)
        if click:
            actions.append(click)

    elif op == "HOVER":
        # UITARS has no explicit hover; approximate as a click on the target.
        click = _build_click_action(step)
        if click:
            actions.append(click)

    elif op == "TYPE":
        # Prefer to click into the target if we can locate it, then type.
        click = _build_click_action(step)
        if click:
            actions.append(click)
        type_val = _build_type_action(step.get("type_action_value"))
        if type_val:
            actions.append(type_val)

    elif op == "SELECT":
        # Many M2W 'SELECT' steps correspond to choosing a value in a combobox.
        # Heuristic: click the control then type the desired value (if provided).
        click = _build_click_action(step)
        if click:
            actions.append(click)
        type_val = _build_type_action(step.get("type_action_value"))
        if type_val:
            actions.append(type_val)

    elif op in ("ENTER", "PRESS ENTER"):
        # Submit via newline per UITARS guidance
        actions.append("type(content='\\n')")
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

