"""
Lightweight concurrency test to ensure no cross-contamination of screenshot history
when evaluating two episodes in parallel. Uses a fake OpenAI client to capture
the images sent per call and verifies each thread only sees its own episode's frames.
"""

import base64
import io
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import pytest

import eval.uitars15_v1 as v1
from eval.episode_loader import load_episode


# Recorder for images observed by the fake client per thread id
_observed_colors = defaultdict(list)


class _FakeResponse:
    class _Msg:
        content = "Action: click(start_box='(10,10)')"

    class _Choice:
        def __init__(self):
            self.message = _FakeResponse._Msg()

    def __init__(self):
        self.choices = [self._Choice()]


class _FakeChat:
    class _Completions:
        def create(self, model, messages, temperature, max_tokens, top_p):
            # Extract all images from the user content, record their top-left pixel color
            colors = []
            for m in messages:
                if m.get("role") != "user":
                    continue
                for part in m.get("content", []):
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        assert url.startswith("data:image/png;base64,")
                        raw = url.split(",", 1)[1]
                        data = base64.b64decode(raw)
                        img = Image.open(io.BytesIO(data))
                        colors.append(img.getpixel((0, 0)))
            _observed_colors[threading.get_ident()].append(tuple(colors))
            return _FakeResponse()

    def __init__(self):
        self.completions = self._Completions()


class _FakeOpenAI:
    def __init__(self, base_url, api_key):
        self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def _patch_openai(monkeypatch):
    # Patch OpenAI client in the agent to avoid network and capture calls
    monkeypatch.setattr(v1, "OpenAI", _FakeOpenAI)
    # Clear recorder before each test
    _observed_colors.clear()


def _make_episode(dir_path: Path, rgb: tuple[int, int, int], steps: int = 3) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    shots = dir_path / "screenshots"
    shots.mkdir(exist_ok=True)
    # Create distinct color images per episode
    for i in range(steps):
        img = Image.new("RGB", (64, 48), color=rgb)
        img.save(shots / f"step_{i}.png")
    traj = []
    for i in range(steps):
        traj.append({
            "confirmed_task": "Task",
            "step_instruction": f"Do step {i}",
            "op": "CLICK",
            "screenshot": str(shots / f"step_{i}.png"),
            "coordinates": [1, 1],
            "bounding_box": [0, 0, 2, 2],
        })
    with open(dir_path / "trajectory.json", "w") as f:
        json.dump(traj, f)
    return dir_path


def _run_episode(ep_dir: Path):
    agent = v1.UITARSAgent(
        model="fake-model",
        runtime_conf={"temperature": 0.0, "max_tokens": 1},
        observation_type="screenshot",
        model_type="qwen25vl",
    )
    for instruction, obs, _ in load_episode(str(ep_dir), instruction_source="step"):
        agent.predict(instruction, obs)


def test_eval_concurrency_no_cross_contamination(tmp_path):
    # Build two distinct episodes with different solid colors
    ep_a = _make_episode(tmp_path / "ep_a", (255, 0, 0))  # red
    ep_b = _make_episode(tmp_path / "ep_b", (0, 0, 255))  # blue

    # Run both concurrently using separate agents (as in the fixed runner)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(_run_episode, ep_a)
        fut_b = ex.submit(_run_episode, ep_b)
        fut_a.result()
        fut_b.result()

    # Each thread should only see its episode's color across all calls
    assert len(_observed_colors) == 2, "Expected two threads recording calls"
    per_thread_unique = []
    for calls in _observed_colors.values():
        # Flatten and get unique set of colors seen by this thread
        colors = set()
        for call_colors in calls:
            colors.update(call_colors)
        per_thread_unique.append(colors)

    # We expect one thread to only see red and the other only blue
    unique_sets = [set(s) for s in per_thread_unique]
    assert any(s == {(255, 0, 0)} for s in unique_sets)
    assert any(s == {(0, 0, 255)} for s in unique_sets)
    # And none should be mixed
    assert all(len(s) == 1 for s in unique_sets), "Mixed colors observed across calls; state leaked between episodes"

