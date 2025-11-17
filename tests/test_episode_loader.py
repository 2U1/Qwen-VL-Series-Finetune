"""
Tests for episode_loader.py
Verifies that trajectory.json + screenshots are correctly transformed into UITARSAgent inputs.
"""

import json
import pytest
from pathlib import Path
from PIL import Image
import io

from eval.episode_loader import load_episode, _resolve_screenshot_path


class TestResolveScreenshotPath:
    """Test screenshot path resolution logic"""
    
    def test_resolve_direct_path(self, tmp_path):
        """Test resolution when JSON path exists directly"""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        
        resolved = _resolve_screenshot_path(str(img_path), tmp_path)
        assert resolved == img_path
    
    def test_resolve_fallback_to_screenshots_dir(self, tmp_path):
        """Test fallback to episode_dir/screenshots/<basename>"""
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        img_path = screenshots_dir / "step_0.png"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        
        # JSON references a non-existent path
        json_path = "outputs/other_location/step_0.png"
        resolved = _resolve_screenshot_path(json_path, tmp_path)
        assert resolved == img_path
    
    def test_resolve_missing_returns_none(self, tmp_path):
        """Test that missing screenshots return None"""
        resolved = _resolve_screenshot_path("nonexistent.png", tmp_path)
        assert resolved is None


class TestLoadEpisode:
    """Test the main load_episode function"""
    
    @pytest.fixture
    def sample_episode(self, tmp_path):
        """Create a minimal valid episode for testing"""
        episode_dir = tmp_path / "episode_123"
        episode_dir.mkdir()
        
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        # Create 3 test screenshots
        for i in range(3):
            img = Image.new("RGB", (800, 600), color=(i*50, i*50, i*50))
            img.save(screenshots_dir / f"step_{i}_click.png")
        
        # Create trajectory.json
        trajectory = [
            {
                "confirmed_task": "Test task for all steps",
                "step_instruction": "Click on button 1",
                "op": "CLICK",
                "screenshot": f"{screenshots_dir}/step_0_click.png",
                "coordinates": [100, 200],
                "bounding_box": [90, 190, 20, 20]
            },
            {
                "confirmed_task": "Test task for all steps",
                "step_instruction": "Type text",
                "op": "TYPE",
                "screenshot": f"{screenshots_dir}/step_1_click.png",
                "type_action_value": "hello world",
                "coordinates": [],
                "bounding_box": [50, 50, 100, 30]
            },
            {
                "confirmed_task": "Test task for all steps",
                "step_instruction": "Click submit",
                "op": "CLICK",
                "screenshot": f"{screenshots_dir}/step_2_click.png",
                "coordinates": [400, 500],
                "bounding_box": [380, 490, 40, 20]
            }
        ]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2)
        
        return episode_dir
    
    def test_load_episode_basic(self, sample_episode):
        """Test basic episode loading"""
        steps = list(load_episode(str(sample_episode), instruction_source="step"))
        
        assert len(steps) == 3, "Should load all 3 steps"
        
        # Check structure of first step
        instruction, obs, metadata = steps[0]
        
        # Verify instruction
        assert instruction == "Click on button 1"
        
        # Verify obs structure
        assert "screenshot" in obs
        assert "accessibility_tree" in obs
        assert isinstance(obs["screenshot"], bytes)
        assert obs["accessibility_tree"] is None
        
        # Verify screenshot is valid PNG
        img = Image.open(io.BytesIO(obs["screenshot"]))
        assert img.size == (800, 600)
        
        # Verify metadata
        assert metadata["op"] == "CLICK"
        assert metadata["step_instruction"] == "Click on button 1"
    
    def test_load_episode_instruction_source_step(self, sample_episode):
        """Test that instruction_source='step' uses step_instruction"""
        steps = list(load_episode(str(sample_episode), instruction_source="step"))
        
        instruction, _, _ = steps[0]
        assert instruction == "Click on button 1"
    
    def test_load_episode_instruction_source_global(self, sample_episode):
        """Test that instruction_source='global' uses confirmed_task"""
        steps = list(load_episode(str(sample_episode), instruction_source="global"))
        
        instruction, _, _ = steps[0]
        assert instruction == "Test task for all steps"
    
    def test_load_episode_maintains_order(self, sample_episode):
        """Test that steps are yielded in JSON order"""
        steps = list(load_episode(str(sample_episode), instruction_source="step"))
        
        instructions = [instr for instr, _, _ in steps]
        expected = ["Click on button 1", "Type text", "Click submit"]
        assert instructions == expected
    
    def test_load_episode_validates_screenshot_count(self, tmp_path):
        """Test that loader validates step count matches screenshot count"""
        episode_dir = tmp_path / "episode_bad"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        # Create only 1 screenshot but 2 steps in JSON
        img = Image.new("RGB", (100, 100))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [
            {
                "confirmed_task": "Task",
                "step_instruction": "Step 1",
                "screenshot": f"{screenshots_dir}/step_0.png"
            },
            {
                "confirmed_task": "Task",
                "step_instruction": "Step 2",
                "screenshot": f"{screenshots_dir}/step_1.png"  # Missing!
            }
        ]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        with pytest.raises(FileNotFoundError, match="could not be resolved"):
            list(load_episode(str(episode_dir)))
    
    def test_load_episode_empty_trajectory_warns(self, tmp_path, capsys):
        """Test that empty trajectory.json produces warning"""
        episode_dir = tmp_path / "episode_empty"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump([], f)
        
        steps = list(load_episode(str(episode_dir)))
        captured = capsys.readouterr()
        
        assert len(steps) == 0
        assert "WARNING: trajectory.json is empty" in captured.out
    
    def test_load_episode_rejects_empty_instruction(self, tmp_path):
        """Test that steps with empty instructions raise error"""
        episode_dir = tmp_path / "episode_bad_instr"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        img = Image.new("RGB", (100, 100))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [
            {
                "screenshot": f"{screenshots_dir}/step_0.png"
                # Missing both step_instruction and confirmed_task
            }
        ]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        with pytest.raises(ValueError, match="empty instruction"):
            list(load_episode(str(episode_dir)))
    
    def test_load_episode_validates_images(self, tmp_path):
        """Test that corrupted images are detected"""
        episode_dir = tmp_path / "episode_corrupt"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        # Create a corrupted "image" file
        corrupt_file = screenshots_dir / "step_0.png"
        with open(corrupt_file, "w") as f:
            f.write("This is not a PNG!")
        
        trajectory = [
            {
                "confirmed_task": "Task",
                "step_instruction": "Step 1",
                "screenshot": str(corrupt_file)
            }
        ]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        with pytest.raises(RuntimeError, match="Failed to read or validate screenshot"):
            list(load_episode(str(episode_dir)))


class TestObservationFormat:
    """Test that observations match UITARSAgent expectations"""
    
    def test_obs_has_required_keys(self, tmp_path):
        """Test obs dict has 'screenshot' and 'accessibility_tree' keys"""
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        img = Image.new("RGB", (640, 480))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [{
            "confirmed_task": "Task",
            "step_instruction": "Do something",
            "screenshot": str(screenshots_dir / "step_0.png")
        }]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        _, obs, _ = next(load_episode(str(episode_dir)))
        
        assert "screenshot" in obs, "obs must have 'screenshot' key"
        assert "accessibility_tree" in obs, "obs must have 'accessibility_tree' key"
        assert isinstance(obs["screenshot"], bytes), "screenshot must be bytes"
        assert obs["accessibility_tree"] is None, "accessibility_tree should be None for screenshot-only"
    
    def test_screenshot_bytes_are_valid_png(self, tmp_path):
        """Test that screenshot bytes can be opened as PIL Image"""
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        # Create an image with specific properties
        img = Image.new("RGB", (1024, 768), color=(255, 0, 0))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [{
            "confirmed_task": "Task",
            "step_instruction": "Do something",
            "screenshot": str(screenshots_dir / "step_0.png")
        }]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        _, obs, _ = next(load_episode(str(episode_dir)))
        
        # Verify bytes can be opened and have correct properties
        loaded_img = Image.open(io.BytesIO(obs["screenshot"]))
        assert loaded_img.size == (1024, 768)
        assert loaded_img.mode == "RGB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

