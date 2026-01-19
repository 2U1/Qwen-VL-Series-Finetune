"""
Integration tests for UITARSAgent with episode loader.
Verifies that prompts are constructed correctly from loaded episodes.
"""

import json
import os
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import io

from eval.episode_loader import load_episode

# Mock mm_agents before importing UITARSAgent
import sys
from unittest.mock import MagicMock
sys.modules['mm_agents'] = MagicMock()
sys.modules['mm_agents.accessibility_tree_wrap'] = MagicMock()
sys.modules['mm_agents.accessibility_tree_wrap.heuristic_retrieve'] = MagicMock()

from eval.uitars15_v1 import UITARSAgent


class TestUITARSAgentIntegration:
    """Test that episode loader output works correctly with UITARSAgent"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client to avoid actual API calls"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """Thought: I will click the button.
Action: click(start_box='<|box_start|>(0.1,0.2)<|box_end|>')"""
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def sample_episode(self, tmp_path):
        """Create a sample episode with realistic data"""
        episode_dir = tmp_path / "test_episode"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        # Create 2 test screenshots
        img1 = Image.new("RGB", (1280, 720), color=(200, 200, 200))
        img1.save(screenshots_dir / "step_0_click.png")
        
        img2 = Image.new("RGB", (1280, 720), color=(150, 150, 150))
        img2.save(screenshots_dir / "step_1_type.png")
        
        trajectory = [
            {
                "confirmed_task": "Book an appointment for passport application",
                "step_instruction": "Click on 'Quick Tools' menuitem",
                "op": "CLICK",
                "screenshot": str(screenshots_dir / "step_0_click.png"),
                "coordinates": [573, 60],
                "bounding_box": [483.734375, 38.5, 180, 44.6875],
                "target_element_type": "menuitem",
                "target_element_text": "Quick Tools"
            },
            {
                "confirmed_task": "Book an appointment for passport application",
                "step_instruction": "Type '60505' in 'City and State, or ZIP Code' textbox",
                "op": "TYPE",
                "screenshot": str(screenshots_dir / "step_1_type.png"),
                "type_action_value": "60505",
                "coordinates": [],
                "bounding_box": [389.5, 379.578125, 505.09375, 44],
                "target_element_type": "textbox",
                "target_element_text": "City and State, or ZIP Code"
            }
        ]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2)
        
        return episode_dir
    
    def test_agent_receives_correct_obs_format(self, sample_episode, mock_openai_client):
        """Test that agent.predict receives obs in correct format"""
        runtime_conf = {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 512,
            "infer_mode": "qwen25vl_normal",
            "prompt_style": "qwen25vl_normal",
            "input_swap": True,
            "language": "English",
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "callusr_tolerance": 1,
            "history_n": 1  # Only current frame for this test
        }
        
        with patch.dict(os.environ, {"DOUBAO_API_URL": "http://test", "DOUBAO_API_KEY": "test"}):
            with patch("eval.uitars15_v1.OpenAI", return_value=mock_openai_client):
                agent = UITARSAgent(
                    model="test-model",
                    runtime_conf=runtime_conf,
                    observation_type="screenshot",
                    model_type="qwen25vl"
                )
                agent.reset()
                
                # Load first step
                instruction, obs, metadata = next(load_episode(str(sample_episode), instruction_source="step"))
                
                # Call predict
                prediction, actions = agent.predict(instruction, obs)
                
                # Verify agent received the data
                assert mock_openai_client.chat.completions.create.called
                call_args = mock_openai_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                
                # Check message structure
                assert len(messages) >= 2  # At least system + user
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                
                # Verify instruction is in the prompt
                user_content = messages[1]["content"]
                text_content = [item for item in user_content if item["type"] == "text"]
                assert len(text_content) > 0
                assert "Click on 'Quick Tools' menuitem" in text_content[0]["text"]
    
    def test_agent_prompt_contains_action_space(self, sample_episode, mock_openai_client):
        """Test that the constructed prompt includes action space"""
        runtime_conf = {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 512,
            "infer_mode": "qwen25vl_normal",
            "prompt_style": "qwen25vl_normal",
            "input_swap": True,
            "language": "English",
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "callusr_tolerance": 1,
            "history_n": 1
        }
        
        with patch.dict(os.environ, {"DOUBAO_API_URL": "http://test", "DOUBAO_API_KEY": "test"}):
            with patch("eval.uitars15_v1.OpenAI", return_value=mock_openai_client):
                agent = UITARSAgent(
                    model="test-model",
                    runtime_conf=runtime_conf,
                    observation_type="screenshot",
                    model_type="qwen25vl"
                )
                agent.reset()
                
                instruction, obs, _ = next(load_episode(str(sample_episode)))
                agent.predict(instruction, obs)
                
                call_args = mock_openai_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                
                # Extract text from user message
                user_text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        for item in msg["content"]:
                            if item["type"] == "text":
                                user_text += item["text"]
                
                # Verify action space is included
                assert "click(start_box=" in user_text
                assert "type(content=" in user_text
                assert "finished" in user_text
    
    def test_agent_receives_image(self, sample_episode, mock_openai_client):
        """Test that agent receives and processes the screenshot"""
        runtime_conf = {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 512,
            "infer_mode": "qwen25vl_normal",
            "prompt_style": "qwen25vl_normal",
            "input_swap": True,
            "language": "English",
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "callusr_tolerance": 1,
            "history_n": 1
        }
        
        with patch.dict(os.environ, {"DOUBAO_API_URL": "http://test", "DOUBAO_API_KEY": "test"}):
            with patch("eval.uitars15_v1.OpenAI", return_value=mock_openai_client):
                agent = UITARSAgent(
                    model="test-model",
                    runtime_conf=runtime_conf,
                    observation_type="screenshot",
                    model_type="qwen25vl"
                )
                agent.reset()
                
                instruction, obs, _ = next(load_episode(str(sample_episode)))
                agent.predict(instruction, obs)
                
                call_args = mock_openai_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                
                # Find image content in messages
                has_image = False
                for msg in messages:
                    for item in msg.get("content", []):
                        if item["type"] == "image_url":
                            has_image = True
                            # Verify it's a base64 encoded image
                            assert "data:image/png;base64," in item["image_url"]["url"]
                
                assert has_image, "Messages should contain an image"
    
    def test_history_window_accumulation(self, sample_episode, mock_openai_client):
        """Test that history_n controls how many images are sent"""
        runtime_conf = {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 512,
            "infer_mode": "qwen25vl_normal",
            "prompt_style": "qwen25vl_normal",
            "input_swap": True,
            "language": "English",
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "callusr_tolerance": 1,
            "history_n": 5
        }
        
        with patch.dict(os.environ, {"DOUBAO_API_URL": "http://test", "DOUBAO_API_KEY": "test"}):
            with patch("eval.uitars15_v1.OpenAI", return_value=mock_openai_client):
                agent = UITARSAgent(
                    model="test-model",
                    runtime_conf=runtime_conf,
                    observation_type="screenshot",
                    model_type="qwen25vl"
                )
                agent.reset()
                
                # Process first step
                steps = list(load_episode(str(sample_episode)))
                agent.predict(*steps[0][:2])
                
                # First call should have 1 image
                first_call = mock_openai_client.chat.completions.create.call_args_list[0]
                first_messages = first_call.kwargs["messages"]
                first_image_count = sum(
                    1 for msg in first_messages 
                    for item in msg.get("content", [])
                    if item["type"] == "image_url"
                )
                assert first_image_count == 1, "First call should have 1 image"
    
    def test_stateless_mode_with_reset(self, sample_episode, mock_openai_client):
        """Test that reset() clears history between steps"""
        runtime_conf = {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 512,
            "infer_mode": "qwen25vl_normal",
            "prompt_style": "qwen25vl_normal",
            "input_swap": True,
            "language": "English",
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "callusr_tolerance": 1,
            "history_n": 1
        }
        
        with patch.dict(os.environ, {"DOUBAO_API_URL": "http://test", "DOUBAO_API_KEY": "test"}):
            with patch("eval.uitars15_v1.OpenAI", return_value=mock_openai_client):
                agent = UITARSAgent(
                    model="test-model",
                    runtime_conf=runtime_conf,
                    observation_type="screenshot",
                    model_type="qwen25vl"
                )
                
                steps = list(load_episode(str(sample_episode)))
                
                # Step 1
                agent.reset()
                instruction1, obs1, _ = steps[0]
                agent.predict(instruction1, obs1)
                
                # Step 2 with reset (stateless)
                agent.reset()
                instruction2, obs2, _ = steps[1]
                agent.predict(instruction2, obs2)
                
                # Check that the second call only has 1 image (no history)
                call_args = mock_openai_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                
                image_count = sum(
                    1 for msg in messages 
                    for item in msg.get("content", [])
                    if item["type"] == "image_url"
                )
                
                assert image_count == 1, "Should only have current image after reset"


class TestPromptConstruction:
    """Test specific aspects of prompt construction"""
    
    def test_instruction_source_affects_prompt(self, tmp_path):
        """Test that instruction_source changes what appears in prompt"""
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        img = Image.new("RGB", (640, 480))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [{
            "confirmed_task": "Global task description",
            "step_instruction": "Step-level instruction",
            "screenshot": str(screenshots_dir / "step_0.png")
        }]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        # Test step instruction
        instruction_step, _, _ = next(load_episode(str(episode_dir), instruction_source="step"))
        assert instruction_step == "Step-level instruction"
        
        # Test global instruction
        instruction_global, _, _ = next(load_episode(str(episode_dir), instruction_source="global"))
        assert instruction_global == "Global task description"
    
    def test_metadata_preserved_for_evaluation(self, tmp_path):
        """Test that metadata contains all original fields for GT comparison"""
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        screenshots_dir = episode_dir / "screenshots"
        screenshots_dir.mkdir()
        
        img = Image.new("RGB", (640, 480))
        img.save(screenshots_dir / "step_0.png")
        
        trajectory = [{
            "confirmed_task": "Task",
            "step_instruction": "Do action",
            "op": "CLICK",
            "coordinates": [100, 200],
            "bounding_box": [90, 190, 20, 20],
            "target_element_type": "button",
            "target_element_text": "Submit",
            "screenshot": str(screenshots_dir / "step_0.png"),
            "custom_field": "custom_value"
        }]
        
        with open(episode_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f)
        
        _, _, metadata = next(load_episode(str(episode_dir)))
        
        # Verify all fields are preserved
        assert metadata["op"] == "CLICK"
        assert metadata["coordinates"] == [100, 200]
        assert metadata["bounding_box"] == [90, 190, 20, 20]
        assert metadata["target_element_type"] == "button"
        assert metadata["target_element_text"] == "Submit"
        assert metadata["custom_field"] == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

