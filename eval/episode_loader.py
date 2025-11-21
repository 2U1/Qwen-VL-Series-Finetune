"""
Load trajectory episodes for UITARSAgent evaluation.

Usage:
    from eval.episode_loader import load_episode
    
    for instruction, obs, metadata in load_episode(episode_dir):
        prediction, actions = agent.predict(instruction, obs)
"""
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional, List, Union, Any

from eval.mind2web_mapping import mind2web_step_to_uitars


def _resolve_screenshot_path(step_screenshot: str, episode_dir: Path) -> Optional[Path]:
    """Resolve screenshot path, trying JSON path first then screenshots/<basename>."""
    candidate = Path(step_screenshot)
    if candidate.is_file():
        return candidate
    screenshots_dir = episode_dir / "screenshots"
    fallback = screenshots_dir / Path(step_screenshot).name
    if fallback.is_file():
        return fallback
    return None


def _target_point_from_step(step: Dict) -> Optional[Tuple[float, float]]:
    """
    Resolve a best-effort target point for the step.
    Preference order:
      1. `coordinates` first (explicit click point)
      2. Center of `bounding_box`
    """
    coords: List[Union[int, float]] = step.get("coordinates") or []
    if len(coords) >= 2:
        try:
            return float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            pass
    bbox: List[Union[int, float]] = step.get("bounding_box") or []
    if len(bbox) >= 4:
        try:
            x, y, w, h = map(float, bbox[:4])
            return x + w / 2.0, y + h / 2.0
        except (TypeError, ValueError):
            return None
    return None


def load_episode(
    episode_dir: str,
    instruction_source: str = "step",
    csv_instruction_lookup: Optional[Any] = None,
) -> Iterator[Tuple[str, Dict, Dict]]:
    """
    Load trajectory from episode_dir and yield (instruction, obs, metadata) per step.
    
    Args:
        episode_dir: Path to outputs/<run>/<episode>/
        instruction_source: "step" for step_instruction, "global" for confirmed_task, 
                           "csv_single_instruction" to use step_instruction from csv,
                           or "csv_multi_element_instruction" to use multi_element_instruction from csv,
        csv_instruction_lookup: Optional dict mapping (task_id, step_index) -> step_instruction, multi_element_instruction, target_coordinates, target_bounding_box.
                               Used when instruction_source="csv_single_instruction" or "csv_multi_element_instruction". The task_id should match
                               the episode folder name.
    
    Yields:
        instruction: Text instruction for the step
        obs: {"screenshot": bytes, "accessibility_tree": None}
        metadata: Original step dict from trajectory.json
    """
    episode_path = Path(episode_dir)

    trajectory_path = episode_path / "trajectory.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory.json not found at: {trajectory_path}")

    with trajectory_path.open("r") as f:
        steps = json.load(f)
    if not isinstance(steps, list):
        raise ValueError("trajectory.json must contain a list of step objects.")
    
    if len(steps) == 0:
        print(f"[episode_loader] WARNING: trajectory.json is empty (0 steps).")
        return

    screenshots_dir = episode_path / "screenshots"
    if not screenshots_dir.is_dir():
        raise FileNotFoundError(f"screenshots directory not found at: {screenshots_dir}")

    present_images = {p.name: p for p in screenshots_dir.glob("*.png")}

    missing_references = []
    resolved_paths = []
    for step in steps:
        if "screenshot" not in step or not step["screenshot"]:
            missing_references.append("<missing screenshot key>")
            resolved_paths.append(None)
            continue
        resolved = _resolve_screenshot_path(step["screenshot"], episode_path)
        if resolved is None:
            missing_references.append(step["screenshot"])
        resolved_paths.append(resolved)

    if missing_references:
        missing_set = "\n  - ".join(str(x) for x in missing_references)
        raise FileNotFoundError(
            "Some screenshots referenced in trajectory.json could not be resolved:\n"
            f"  - {missing_set}\n"
            f"Episode dir checked: {episode_path}"
        )

    if len(steps) != len(resolved_paths):
        raise AssertionError(
            f"Mismatch: steps={len(steps)} vs resolved screenshots={len(resolved_paths)}"
        )

    # Warn about unreferenced images
    referenced_basenames = {Path(p).name for p in resolved_paths if p is not None}
    extra_images = sorted(set(present_images.keys()) - referenced_basenames)
    if len(extra_images) > 0:
        print(
            f"[episode_loader] Warning: {len(extra_images)} images present but not referenced by JSON.\n"
            f"  Examples: {extra_images[:5]}"
        )

    # Get task_id from episode folder name for CSV lookup
    task_id = episode_path.name if csv_instruction_lookup is not None else None
    
    for step_idx, (step, img_path) in enumerate(zip(steps, resolved_paths)):
        if instruction_source == "csv_single_instruction":
            if csv_instruction_lookup is None:
                raise ValueError("csv_instruction_lookup must be provided when instruction_source='csv'")
            if task_id is None:
                raise ValueError("Cannot determine task_id from episode_dir for CSV lookup")
            # Look up instruction from CSV using (task_id, step_index)
            instruction = csv_instruction_lookup.get((task_id, step_idx)).get("step_instruction")
            if instruction is None:
                # Fallback to trajectory.json if CSV lookup fails
                instruction = step.get("step_instruction", "") or step.get("confirmed_task", "")
                if instruction:
                    print(f"[episode_loader] WARNING: No CSV instruction for task_id={task_id}, step={step_idx}, using trajectory.json")
        elif instruction_source == "csv_multi_element_instruction":
            if csv_instruction_lookup is None:
                raise ValueError("csv_instruction_lookup must be provided when instruction_source='csv_multi_element_instruction'")
            if task_id is None:
                raise ValueError("Cannot determine task_id from episode_dir for CSV lookup")
            # Look up instruction from CSV using (task_id, step_index)
            instruction = csv_instruction_lookup.get((task_id, step_idx)).get("multi_element_instruction")
            if instruction is None:
                # Fallback to trajectory.json if CSV lookup fails
                instruction = step.get("multi_element_instruction", "") or step.get("confirmed_task", "")
                if instruction:
                    print(f"[episode_loader] WARNING: No CSV instruction for task_id={task_id}, step={step_idx}, using trajectory.json")
        elif instruction_source == "step":
            instruction = step.get("step_instruction", "") or step.get("confirmed_task", "")
        elif instruction_source == "global":
            instruction = step.get("confirmed_task", "") or step.get("step_instruction", "")
        else:
            raise ValueError("instruction_source must be 'step', 'global', 'csv_single_instruction', or 'csv_multi_element_instruction'")
        
        if not instruction or not instruction.strip():
            raise ValueError(
                f"Step {step_idx} has empty instruction. "
                f"Both 'step_instruction' and 'confirmed_task' are missing or empty, "
                f"and CSV lookup (if used) returned None."
            )

        try:
            with img_path.open("rb") as f:
                screenshot_bytes = f.read()
            from PIL import Image
            Image.open(img_path).verify()
        except Exception as e:
            raise RuntimeError(
                f"Step {step_idx}: Failed to read or validate screenshot {img_path}: {e}"
            )
        
        obs = {"screenshot": screenshot_bytes, "accessibility_tree": None}

        annotated_step = dict(step)
        try:
            uitars_actions = mind2web_step_to_uitars(step)
        except Exception as e:
            raise RuntimeError(f"Step {step_idx}: Failed to convert to UITARS actions: {e}") from e
        annotated_step["uitars_actions"] = uitars_actions
        if instruction_source == "csv_single_instruction" or instruction_source == "csv_multi_element_instruction":
            annotated_step["target_coordinates"] = csv_instruction_lookup.get((task_id, step_idx)).get("target_coordinates")
            annotated_step["target_bounding_box"] = csv_instruction_lookup.get((task_id, step_idx)).get("target_bounding_box")
        else:
            target_point = _target_point_from_step(step)
            if target_point is not None:
                annotated_step["target_point"] = target_point

        annotated_step["screenshot_path"] = str(img_path)

        yield instruction, obs, annotated_step


