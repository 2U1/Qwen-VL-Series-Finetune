import pytest

from eval.mind2web_mapping import mind2web_step_to_uitars, _format_uitars_box


def test_click_maps_to_click_with_coordinates():
    step = {
        "op": "CLICK",
        "coordinates": [573, 60],
        "bounding_box": [483.734375, 38.5, 180, 44.6875],
    }
    actions = mind2web_step_to_uitars(step)
    assert actions == [f"click(start_box='{_format_uitars_box(573, 60)}')"]


def test_click_uses_bbox_when_no_coordinates():
    step = {
        "op": "CLICK",
        "coordinates": [],
        "bounding_box": [100, 200, 20, 20],  # center -> (110, 210)
    }
    actions = mind2web_step_to_uitars(step)
    assert actions == [f"click(start_box='{_format_uitars_box(110, 210)}')"]


def test_type_maps_to_click_then_type():
    step = {
        "op": "TYPE",
        "coordinates": [],
        "bounding_box": [389.5, 379.578125, 505.09375, 44],  # center -> (642, 401)
        "type_action_value": "hello world",
    }
    actions = mind2web_step_to_uitars(step)
    assert len(actions) == 2
    assert actions[0].startswith("click(")
    assert "type(content='hello world')" == actions[1]

def test_hover_maps_to_click():
    step = {
        "op": "HOVER",
        "coordinates": [200, 300],
        "bounding_box": [0, 0, 10, 10],
    }
    actions = mind2web_step_to_uitars(step)
    assert actions == [f"click(start_box='{_format_uitars_box(200, 300)}')"]

def test_press_enter_maps_to_type_newline():
    step = {
        "op": "ENTER",
    }
    actions = mind2web_step_to_uitars(step)
    assert actions == ["type(content='\\n')"]


def test_select_maps_to_click_then_type_when_value_present():
    step = {
        "op": "SELECT",
        "coordinates": [619, 419],
        "bounding_box": [394.5, 400.765625, 449.578125, 38.15625],
        "type_action_value": "New Passport Only",
    }
    actions = mind2web_step_to_uitars(step)
    assert actions == [
        f"click(start_box='{_format_uitars_box(619, 419)}')",
        "type(content='New Passport Only')",
    ]


