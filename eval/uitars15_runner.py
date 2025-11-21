"""
Run UITARSAgent evaluation on recorded trajectory episodes.

Usage (single episode):
    python -m eval.uitars15_runner \\
        --episode_dir /abs/path/to/outputs/run_X/<episode_id> \\
        --model your-model

Usage (entire run directory of episodes):
    python -m eval.uitars15_runner \\
        --run_dir /abs/path/to/outputs/run_X \\
        --model your-model

By default, per-step LLM predictions are written to
`./uitars_eval_<timestamp>/uitars_predictions/<episode_id>.jsonl` in the
current working directory. Metrics and summary files are similarly written to
that timestamped directory. You can also optionally provide `--output_jsonl`
to collect all steps into a single aggregated JSONL file at an arbitrary path.

Set environment variables before running:
    export DOUBAO_API_URL="https://your-endpoint.com/v1"
    export DOUBAO_API_KEY="your-api-key"
"""
import argparse
import ast
import json
import os
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
from tqdm import tqdm

from eval.episode_loader import load_episode
from eval.uitars15_v1 import UITARSAgent, compute_step_metrics


logger = logging.getLogger(__name__)


STEP_METRIC_KEYS = ("action_str_em", "hit_box_accuracy", "bbox_center_mse")


def build_runtime_conf(args: argparse.Namespace) -> Dict[str, Any]:
    """Build runtime_conf dict for UITARSAgent."""
    runtime_conf: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "infer_mode": args.infer_mode,
        "prompt_style": args.prompt_style,
        "input_swap": args.input_swap,
        "language": args.language,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "callusr_tolerance": args.callusr_tolerance,
        "history_n": args.history_n,
    }
    if args.seed is not None:
        runtime_conf["seed"] = args.seed
    return runtime_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UITARSAgent over recorded episode(s).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--episode_dir", type=str, help="Path to outputs/<run>/<episode> directory.")
    input_group.add_argument("--run_dir", type=str, help="Path to a run directory containing episode subdirectories.")
    input_group.add_argument("--base_dir", type=str, help="Path to base directory containing multiple run directories.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root directory for metrics, predictions, and summary. "
            "Defaults to ./uitars_eval_<timestamp> under the current working directory."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help=(
            "If set together with --run_dir, evaluate at most this many episodes "
            "from the sorted list of available episodes."
        ),
    )
    parser.add_argument("--model", type=str, required=True, help="Model name/id for the OpenAI-compatible endpoint.")
    parser.add_argument(
        "--instruction_source",
        type=str,
        choices=["step", "global", "csv", "csv_single_instruction", "csv_multi_element_instruction"],
        default="step"
    )
    parser.add_argument(
        "--csv_instructions",
        type=str,
        default=None,
        help="Path to CSV file with instructions. Required when --instruction_source=csv. "
             "CSV should have columns: task_id, step_index, step_instruction"
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        choices=["screenshot", "screenshot_a11y_tree"],
        default="screenshot",
    )
    parser.add_argument("--infer_mode", type=str, default="qwen25vl_normal")
    parser.add_argument("--prompt_style", type=str, default="qwen25vl_normal")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--history_n",
        type=int,
        default=5,
        help="How many past screenshots to include. Default is 5.",
    )
    parser.add_argument(
        "--reset_each_step",
        action="store_true",
        help="If set, reset agent state before each step (stateless per-step prompts).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation. If not set, generation will be non-deterministic.",
    )
    parser.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=100 * 28 * 28)
    parser.add_argument("--input_swap", action="store_true", help="Use clipboard paste for typing.")
    parser.add_argument("--no-input_swap", dest="input_swap", action="store_false")
    parser.set_defaults(input_swap=True)
    parser.add_argument("--callusr_tolerance", type=int, default=3)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional path to write per-step predictions as JSONL (aggregated if --run_dir is provided).",
    )
    parser.add_argument(
        "--metrics_jsonl",
        type=str,
        default=None,
        help="Optional path to write per-step metrics as JSONL.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help=(
            "Optional path to write per-episode summary metrics as a single JSON file. "
            "Each entry contains episode id, number of steps, and averaged metrics."
        ),
    )

    return parser.parse_args()


def _iter_episode_dirs(run_dir: Path):
    """Yield episode subdirectories under run_dir that contain trajectory.json and screenshots/."""
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Invalid run_dir: {run_dir}")
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        traj = child / "trajectory.json"
        shots = child / "screenshots"
        if traj.is_file() and shots.is_dir():
            yield child


def _iter_run_dirs(base_dir: Path):
    """Yield run directories that contain task subdirectories."""
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Invalid base_dir: {base_dir}")
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        # Check if this directory contains task folders (subdirectories with screenshots/)
        has_task_folders = False
        for task_dir in child.iterdir():
            if task_dir.is_dir() and (task_dir / "screenshots").is_dir():
                has_task_folders = True
                break
        if has_task_folders:
            yield child


def _load_csv_instructions(csv_path: str) -> Dict[Tuple[str, int], str]:
    """
    Load CSV and create a lookup dict mapping (task_id, step_index) -> step_instruction.
    
    Args:
        csv_path: Path to CSV file with columns: task_id, step_index, step_instruction
    
    Returns:
        Dict mapping (task_id, step_index) -> step_instruction
    """
    df = pd.read_csv(csv_path)
    required_cols = ["task_id", "step_index", "step_instruction", "multi_element_instruction", "target_coordinates", "target_bounding_box"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    lookup = {}
    for _, row in df.iterrows():
        task_id = str(row["task_id"]).strip()
        step_index = int(row["step_index"])
        instruction = str(row["step_instruction"]).strip()
        multi_element_instruction = str(row["multi_element_instruction"]).strip()
        target_coordinates = list(ast.literal_eval(str(row["target_coordinates"]).strip()))
        target_bounding_box = list(ast.literal_eval(str(row["target_bounding_box"]).strip()))
        lookup[(task_id, step_index)] = {
            "step_instruction": instruction,
            "multi_element_instruction": multi_element_instruction,
            "target_coordinates": target_coordinates,
            "target_bounding_box": target_bounding_box,
        }
    
    return lookup


def main() -> None:
    args = parse_args()

    # Default output locations live under a timestamped directory in the
    # current working directory unless explicitly overridden by CLI flags.
    cwd = Path.cwd()
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = cwd / f"uitars_eval_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Enable default aggregated outputs if not explicitly provided.
    if args.metrics_jsonl is None:
        args.metrics_jsonl = str(output_root / "uitars_metrics.jsonl")
    if args.summary_json is None:
        args.summary_json = str(output_root / "uitars_summary.json")

    # Configure basic logging if the root logger has no handlers yet.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

    # Load CSV instructions if using CSV source
    csv_instruction_lookup: Optional[Dict[Tuple[str, int], str]] = None
    if args.instruction_source == "csv_single_instruction" or args.instruction_source == "csv_multi_element_instruction":
        if args.csv_instructions is None:
            raise ValueError("--csv_instructions is required when --instruction_source=csv_single_instruction or --instruction_source=csv_multi_element_instruction")
        logger.info("Loading instructions from CSV: %s", args.csv_instructions)
        csv_instruction_lookup = _load_csv_instructions(args.csv_instructions)
        logger.info("Loaded %d entries from CSV", len(csv_instruction_lookup))

    runtime_conf = build_runtime_conf(args)
    agent = UITARSAgent(
        model=args.model,
        runtime_conf=runtime_conf,
        observation_type=args.observation_type,
        model_type="qwen25vl",
    )

    episode_summaries: List[Dict[str, Any]] = []
    # Global aggregators across all processed episodes/steps.
    global_metric_totals: Dict[str, float] = defaultdict(float)
    global_metric_counts: Dict[str, int] = defaultdict(int)
    global_num_steps: int = 0
    global_num_episodes: int = 0

    jsonl_file: Optional[Any] = None
    if args.output_jsonl is not None:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = out_path.open("w")
    
    metrics_file: Optional[Any] = None
    if args.metrics_jsonl is not None:
        metrics_path = Path(args.metrics_jsonl)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w")

    # Default directory to store per-episode LLM predictions.
    # This is rooted in the output_root directory by default so that
    # running the script from any location keeps all artifacts for a run
    # grouped together, unless paths are explicitly redirected via CLI flags.
    predictions_root = output_root / "uitars_predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    def evaluate_one_episode(ep_dir: Path) -> None:
        nonlocal jsonl_file
        nonlocal metrics_file
        nonlocal episode_summaries
        nonlocal global_metric_totals
        nonlocal global_metric_counts
        nonlocal global_num_steps
        nonlocal global_num_episodes
        nonlocal predictions_root

        logger.info("Evaluating episode '%s'", ep_dir.name)
        agent.reset()

        # Aggregators
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)
                
        # Collect step-level data for summary
        step_data: List[Dict[str, Any]] = []

        episode_pred_path = predictions_root / f"{ep_dir.name}.jsonl"
        episode_pred_file: Optional[Any] = episode_pred_path.open("w", encoding="utf-8")

        try:
            step_index = -1
            step_iterator = load_episode(
                str(ep_dir), 
                instruction_source=args.instruction_source,
                csv_instruction_lookup=csv_instruction_lookup
            )
            for step_index, (instruction, obs, metadata) in enumerate(
                tqdm(step_iterator, desc=f"Steps [{ep_dir.name}]", unit="step")
            ):
                if args.reset_each_step:
                    agent.reset()

                try:
                    prediction, actions = agent.predict(instruction, obs)
                except Exception as e:
                    logger.error(
                        "Error during prediction: episode=%s step=%d error=%s",
                        ep_dir.name,
                        step_index,
                        e,
                    )
                    record = {
                        "episode": ep_dir.name,
                        "step_index": step_index,
                        "instruction": instruction,
                        "action_uid": metadata.get("action_uid"),
                        "error": str(e),
                        "metadata": metadata,
                    }
                    if jsonl_file is not None:
                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        jsonl_file.flush()
                    episode_pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    episode_pred_file.flush()
                    raise
                
                # Compute metrics per step
                try:
                    metrics = compute_step_metrics(
                        prediction_text=prediction,
                        screenshot_bytes=obs["screenshot"],
                        metadata=metadata,
                        model_type="qwen25vl",
                        max_pixels=args.max_pixels,
                        min_pixels=args.min_pixels
                    )
                except Exception as _:
                    metrics = {key: None for key in STEP_METRIC_KEYS}

                is_terminal = False
                if prediction == "client error" or actions in [["DONE"], ["FAIL"]]:
                    is_terminal = True
                    if prediction == "client error":
                        logger.warning(
                            "Client error at episode=%s step=%d", ep_dir.name, step_index
                        )
                    elif actions == ["DONE"]:
                        logger.info(
                            "Task completed at episode=%s step=%d",
                            ep_dir.name,
                            step_index,
                        )
                    elif actions == ["FAIL"]:
                        logger.info(
                            "Task failed at episode=%s step=%d",
                            ep_dir.name,
                            step_index,
                        )
                metric_parts = []
                for key in STEP_METRIC_KEYS:
                    value = metrics.get(key)
                    if value is None:
                        metric_parts.append(f"{key}=NA")
                    else:
                        metric_parts.append(f"{key}={value:.3f}")
                logger.info(
                    "episode=%s step=%d metrics=%s",
                    ep_dir.name,
                    step_index,
                    " ".join(metric_parts),
                )

                record = {
                    "episode": ep_dir.name,
                    "step_index": step_index,
                    "instruction": instruction,
                    "action_uid": metadata.get("action_uid"),
                    "prediction": prediction,
                    "predicted_actions": actions,
                    "metadata": metadata,
                }
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                episode_pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                episode_pred_file.flush()

                if metrics_file is not None:
                    mrec = {
                        "episode": ep_dir.name,
                        "step_index": step_index,
                        "action_uid": metadata.get("action_uid"),
                        "op": metadata.get("op"),
                        "ground_truth_actions": metadata.get("uitars_actions", []),
                        "prediction": prediction,
                        "predicted_actions": actions,
                        "screenshot_path": metadata.get("screenshot_path"),
                        "metrics": metrics,
                    }
                    metrics_file.write(json.dumps(mrec, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                for key, value in metrics.items():
                    if value is None:
                        continue
                    metric_totals[key] += float(value)
                    metric_counts[key] += 1
                    global_metric_totals[key] += float(value)
                    global_metric_counts[key] += 1

                # Count every processed step once for global statistics,
                # regardless of how many metrics are defined for it.
                global_num_steps += 1
                                
                # Collect step data for summary
                step_data.append({
                    "step_index": step_index,
                    "instruction": instruction,
                    "ground_truth": metadata.get("uitars_actions", []),
                    "prediction": prediction,
                    "screenshot_path": metadata.get("screenshot_path"),
                })

                if is_terminal:
                    logger.info(
                        "Stopping evaluation for episode=%s due to terminal state.",
                        ep_dir.name,
                    )
                    break
        finally:
            episode_pred_file.close()

        num_steps = step_index + 1 if step_index >= 0 else 0
        summary_parts = []
        for key in STEP_METRIC_KEYS:
            count = metric_counts.get(key, 0)
            if count == 0:
                summary_parts.append(f"{key}=NA")
            else:
                avg = metric_totals[key] / count
                summary_parts.append(f"{key}={avg:.3f} (n={count})")
        logger.info(
            "Completed %d steps for episode=%s. %s",
            num_steps,
            ep_dir.name,
            " ".join(summary_parts),
        )

        # Record a compact JSON-serializable summary for this episode.
        episode_summary: Dict[str, Any] = {
            "episode": ep_dir.name,
            "num_steps": num_steps,
            "metrics": {},
            "steps": step_data,
        }
        for key in STEP_METRIC_KEYS:
            count = metric_counts.get(key, 0)
            if count == 0:
                episode_summary["metrics"][key] = {"mean": None, "count": 0}
            else:
                avg = metric_totals[key] / count
                episode_summary["metrics"][key] = {"mean": avg, "count": count}
        episode_summaries.append(episode_summary)
        global_num_episodes += 1

    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
        if not episode_dir.is_dir():
            raise NotADirectoryError(f"Invalid episode_dir: {episode_dir}")
        evaluate_one_episode(episode_dir)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        episode_dirs = list(_iter_episode_dirs(run_dir))
        logger.info("Found %d episode(s) under run_dir=%s", len(episode_dirs), run_dir)

        # Optionally limit the number of episodes evaluated.
        if args.max_episodes is not None:
            if args.max_episodes < 0:
                raise ValueError("--max_episodes must be non-negative if provided.")
            episode_dirs = episode_dirs[: args.max_episodes]
            logger.info(
                "Restricting evaluation to first %d episode(s) after sorting.",
                len(episode_dirs),
            )
        for ep in tqdm(episode_dirs, desc="Episodes", unit="episode"):
            evaluate_one_episode(ep)
    elif args.base_dir:
        base_dir = Path(args.base_dir)
        run_dirs = list(_iter_run_dirs(base_dir))
        logger.info("Found %d run directory(ies) under base_dir=%s", len(run_dirs), base_dir)
        
        # Process each run directory separately with its own output directory
        for run_dir in tqdm(run_dirs, desc="Run directories", unit="run"):
            logger.info("=" * 80)
            logger.info("Processing run directory: %s", run_dir.name)
            logger.info("=" * 80)
            
            # Create output directory named after the run directory
            if args.output_root is not None:
                run_output_root = Path(args.output_root) / run_dir.name
            else:
                run_output_root = cwd / run_dir.name
            run_output_root.mkdir(parents=True, exist_ok=True)
            
            # Set up per-run output paths
            run_predictions_root = run_output_root / "uitars_predictions"
            run_predictions_root.mkdir(parents=True, exist_ok=True)
            
            # Update predictions_root for this run
            original_predictions_root = predictions_root
            predictions_root = run_predictions_root
            
            # Set up per-run metrics and summary files
            run_metrics_jsonl = run_output_root / "uitars_metrics.jsonl"
            run_summary_json = run_output_root / "uitars_summary.json"
            
            # Reset aggregators for this run (each run is independent)
            metrics_file = run_metrics_jsonl.open("w")
            episode_summaries = []
            global_metric_totals = defaultdict(float)
            global_metric_counts = defaultdict(int)
            global_num_steps = 0
            global_num_episodes = 0
            
            episode_dirs = list(_iter_episode_dirs(run_dir))
            logger.info("Found %d episode(s) in run_dir=%s", len(episode_dirs), run_dir.name)
            
            if len(episode_dirs) == 0:
                logger.warning("No episodes found in run_dir=%s, skipping", run_dir.name)
                metrics_file.close()
                continue
            
            # Optionally limit the number of episodes evaluated per run directory
            if args.max_episodes is not None:
                if args.max_episodes < 0:
                    raise ValueError("--max_episodes must be non-negative if provided.")
                episode_dirs = episode_dirs[: args.max_episodes]
                logger.info(
                    "Restricting evaluation to first %d episode(s) in this run directory.",
                    len(episode_dirs),
                )
            
            for ep in tqdm(episode_dirs, desc=f"Episodes [{run_dir.name}]", unit="episode"):
                evaluate_one_episode(ep)
            
            # Write summary for this run
            metrics_file.close()
            if run_summary_json is not None:
                summary_path = Path(run_summary_json)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                # Build a global aggregate over all processed steps for this run.
                run_global_metrics: Dict[str, Any] = {
                    "num_episodes": global_num_episodes,
                    "num_steps": global_num_steps,
                    "metrics": {},
                }
                for key in STEP_METRIC_KEYS:
                    count = global_metric_counts.get(key, 0)
                    if count == 0:
                        run_global_metrics["metrics"][key] = {"mean": None, "count": 0}
                    else:
                        avg = global_metric_totals[key] / count
                        run_global_metrics["metrics"][key] = {"mean": avg, "count": count}

                payload: Dict[str, Any] = {
                    "episodes": episode_summaries,
                    "global": run_global_metrics,
                }

                with summary_path.open("w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                logger.info(
                    "Wrote summary metrics (per-episode and global) to %s", summary_path
                )
            
            # Restore predictions_root for next iteration (though it will be overwritten anyway)
            predictions_root = original_predictions_root
            
            logger.info("Completed processing run directory: %s", run_dir.name)
    
    if jsonl_file is not None:
        jsonl_file.close()
    if metrics_file is not None:
        metrics_file.close()

    # Optionally write per-episode and global summary metrics to a compact JSON file.
    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Build a global aggregate over all processed steps.
        global_metrics: Dict[str, Any] = {
            "num_episodes": global_num_episodes,
            "num_steps": global_num_steps,
            "metrics": {},
        }
        for key in STEP_METRIC_KEYS:
            count = global_metric_counts.get(key, 0)
            if count == 0:
                global_metrics["metrics"][key] = {"mean": None, "count": 0}
            else:
                avg = global_metric_totals[key] / count
                global_metrics["metrics"][key] = {"mean": avg, "count": count}

        payload: Dict[str, Any] = {
            "episodes": episode_summaries,
            "global": global_metrics,
        }

        with summary_path.open("w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(
            "Wrote summary metrics (per-episode and global) to %s", summary_path
        )


if __name__ == "__main__":
    main()


