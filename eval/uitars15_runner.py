"""
Run UITARSAgent evaluation on recorded trajectory episodes.

Usage (single episode):
    python -m eval.uitars15_runner \\
        --episode_dir /abs/path/to/outputs/run_X/<episode_id> \\
        --model your-model \\
        --output_jsonl /abs/path/to/results.jsonl

Usage (entire run directory of episodes):
    python -m eval.uitars15_runner \\
        --run_dir /abs/path/to/outputs/run_X \\
        --model your-model \\
        --output_jsonl /abs/path/to/results.jsonl

Set environment variables before running:
    export DOUBAO_API_URL="https://your-endpoint.com/v1"
    export DOUBAO_API_KEY="your-api-key"
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from eval.episode_loader import load_episode
from eval.uitars15_v1 import UITARSAgent, compute_step_metrics


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

    runtime_conf = build_runtime_conf(args)
    agent = UITARSAgent(
        model=args.model,
        runtime_conf=runtime_conf,
        observation_type=args.observation_type,
        model_type="qwen25vl",
    )

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

    def evaluate_one_episode(ep_dir: Path) -> None:
        nonlocal jsonl_file
        nonlocal metrics_file
        agent.reset()

        # Aggregators
        steps = 0
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)

        step_index = 0
        for instruction, obs, metadata in load_episode(
            str(ep_dir), instruction_source=args.instruction_source
        ):
            if args.reset_each_step:
                agent.reset()

            try:
                prediction, actions = agent.predict(instruction, obs)
            except Exception as e:
                print(f"[runner] ERROR episode={ep_dir.name} step={step_index}: {e}")
                if jsonl_file is not None:
                    record = {
                        "episode": ep_dir.name,
                        "step_index": step_index,
                        "instruction": instruction,
                        "error": str(e),
                        "metadata": metadata,
                    }
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
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
                    print(f"[runner] WARNING: Client error at step={step_index}")
                elif actions == ["DONE"]:
                    print(f"[runner] Task completed at step={step_index}")
                elif actions == ["FAIL"]:
                    print(f"[runner] Task failed at step={step_index}")
            print(f"[runner] episode={ep_dir.name} step={step_index}")
            print(f"[runner] instruction={instruction}")
            metric_parts = []
            for key in STEP_METRIC_KEYS:
                value = metrics.get(key)
                if value is None:
                    metric_parts.append(f"{key}=NA")
                else:
                    metric_parts.append(f"{key}={value:.3f}")
            print(f"[runner] prediction={prediction}")
            print(f"[runner] actions={actions}")
            print(f"[runner] metrics {' '.join(metric_parts)}")
            if jsonl_file is not None:
                record = {
                    "episode": ep_dir.name,
                    "step_index": step_index,
                    "instruction": instruction,
                    "prediction": prediction,
                    "actions": actions,
                    "metadata": metadata,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()
            if metrics_file is not None:
                mrec = {
                    "episode": ep_dir.name,
                    "step_index": step_index,
                    "op": metadata.get("op"),
                    "ground_truth_actions": metadata.get("uitars_actions", []),
                    "metrics": metrics,
                }
                metrics_file.write(json.dumps(mrec, ensure_ascii=False) + "\n")
                metrics_file.flush()

            steps += 1
            for key, value in metrics.items():
                if value is None:
                    continue
                metric_totals[key] += float(value)
                metric_counts[key] += 1
            step_index += 1
            if is_terminal:
                print(f"[runner] Stopping evaluation due to terminal state.")
                break

        summary_parts = []
        for key in STEP_METRIC_KEYS:
            count = metric_counts.get(key, 0)
            if count == 0:
                summary_parts.append(f"{key}=NA")
            else:
                avg = metric_totals[key] / count
                summary_parts.append(f"{key}={avg:.3f} (n={count})")
        print(
            f"[runner] Completed {step_index} steps for episode={ep_dir.name}. "
            + " ".join(summary_parts)
        )

    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
        if not episode_dir.is_dir():
            raise NotADirectoryError(f"Invalid episode_dir: {episode_dir}")
        evaluate_one_episode(episode_dir)
    else:
        run_dir = Path(args.run_dir)
        for ep in _iter_episode_dirs(run_dir):
            evaluate_one_episode(ep)
    
    if jsonl_file is not None:
        jsonl_file.close()
    if metrics_file is not None:
        metrics_file.close()


if __name__ == "__main__":
    main()


