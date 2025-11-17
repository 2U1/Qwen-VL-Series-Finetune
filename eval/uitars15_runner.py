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
`<run_dir>/uitars_predictions/<episode_id>.jsonl`. You can also optionally
provide `--output_jsonl` to collect all steps into a single aggregated JSONL
file at an arbitrary path.

Set environment variables before running:
    export DOUBAO_API_URL="https://your-endpoint.com/v1"
    export DOUBAO_API_KEY="your-api-key"
"""
import argparse
import json
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, List

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
    }
    if args.history_n is not None:
        runtime_conf["history_n"] = args.history_n
    return runtime_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UITARSAgent over recorded episode(s).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--episode_dir", type=str, help="Path to outputs/<run>/<episode> directory.")
    input_group.add_argument("--run_dir", type=str, help="Path to a run directory containing episode subdirectories.")
    parser.add_argument("--model", type=str, required=True, help="Model name/id for the OpenAI-compatible endpoint.")
    parser.add_argument("--instruction_source", type=str, choices=["step", "global"], default="step")
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
        default=None,
        help="How many past screenshots to include. Default None means use class default (5).",
    )
    parser.add_argument(
        "--reset_each_step",
        action="store_true",
        help="If set, reset agent state before each step (stateless per-step prompts).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=100 * 28 * 28)
    parser.add_argument("--input_swap", action="store_true", help="Use clipboard paste for typing.")
    parser.add_argument("--no-input_swap", dest="input_swap", action="store_false")
    parser.set_defaults(input_swap=True)
    parser.add_argument("--callusr_tolerance", type=int, default=1)
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


def main() -> None:
    args = parse_args()

    # Derive a base directory for default output locations.
    if args.run_dir:
        _base_dir = Path(args.run_dir)
    else:
        _base_dir = Path(args.episode_dir).parent

    # Enable default aggregated outputs if not explicitly provided.
    if args.metrics_jsonl is None:
        args.metrics_jsonl = str(_base_dir / "uitars_metrics.jsonl")
    if args.summary_json is None:
        args.summary_json = str(_base_dir / "uitars_summary.json")

    # Configure basic logging if the root logger has no handlers yet.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

    runtime_conf = build_runtime_conf(args)
    agent = UITARSAgent(
        model=args.model,
        runtime_conf=runtime_conf,
        observation_type=args.observation_type,
        model_type="qwen25vl",
    )

    episode_summaries: List[Dict[str, Any]] = []

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
    if args.run_dir:
        predictions_root = Path(args.run_dir) / "uitars_predictions"
    else:
        episode_dir_for_output = Path(args.episode_dir)
        predictions_root = episode_dir_for_output.parent / "uitars_predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    def evaluate_one_episode(ep_dir: Path) -> None:
        nonlocal jsonl_file
        nonlocal metrics_file
        nonlocal episode_summaries

        logger.info("Evaluating episode '%s'", ep_dir.name)
        agent.reset()

        # Aggregators
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)

        episode_pred_path = predictions_root / f"{ep_dir.name}.jsonl"
        episode_pred_file: Optional[Any] = episode_pred_path.open("w", encoding="utf-8")

        try:
            step_index = -1
            step_iterator = load_episode(
                str(ep_dir), instruction_source=args.instruction_source
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
                    "actions": actions,
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
                        "metrics": metrics,
                    }
                    metrics_file.write(json.dumps(mrec, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                for key, value in metrics.items():
                    if value is None:
                        continue
                    metric_totals[key] += float(value)
                    metric_counts[key] += 1
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
        }
        for key in STEP_METRIC_KEYS:
            count = metric_counts.get(key, 0)
            if count == 0:
                episode_summary["metrics"][key] = {"mean": None, "count": 0}
            else:
                avg = metric_totals[key] / count
                episode_summary["metrics"][key] = {"mean": avg, "count": count}
        episode_summaries.append(episode_summary)

    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
        if not episode_dir.is_dir():
            raise NotADirectoryError(f"Invalid episode_dir: {episode_dir}")
        evaluate_one_episode(episode_dir)
    else:
        run_dir = Path(args.run_dir)
        episode_dirs = list(_iter_episode_dirs(run_dir))
        logger.info("Found %d episode(s) under run_dir=%s", len(episode_dirs), run_dir)
        for ep in tqdm(episode_dirs, desc="Episodes", unit="episode"):
            evaluate_one_episode(ep)
    
    if jsonl_file is not None:
        jsonl_file.close()
    if metrics_file is not None:
        metrics_file.close()

    # Optionally write per-episode summary metrics to a compact JSON file.
    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(episode_summaries, f, ensure_ascii=False, indent=2)
        logger.info("Wrote summary metrics to %s", summary_path)


if __name__ == "__main__":
    main()


