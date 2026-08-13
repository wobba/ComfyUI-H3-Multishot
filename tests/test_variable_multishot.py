import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_schedule():
    utils = load_module("h3_multishot_utils")
    assert utils._parse_frame_schedule("", 3, 243) == [243, 243, 243]
    assert utils._parse_frame_schedule("124||362", 3, 243) == [124, 243, 362]
    assert utils._parse_frame_schedule(
        "[124, null, 362]", 3, 243
    ) == [124, 243, 362]
    assert utils._parse_frame_schedule("124", 3, 243) == [124, 243, 243]
    assert utils._parse_frame_schedule(
        "123", 1, 243, lambda _frames: 124
    ) == [124]
    prompts, frames = utils._resolve_segment_frames(
        [
            "frame_count: 124\nFirst prompt",
            "Second prompt",
            "frame_count: 362\nThird prompt",
        ],
        "",
        243,
    )
    assert frames == [124, 243, 362]
    assert prompts == ["First prompt", "Second prompt", "Third prompt"]
    prompts, frames = utils._resolve_segment_frames(
        ["frame_count: 124\nFirst prompt", "Second prompt"],
        "362|175",
        243,
    )
    assert frames == [124, 175]


def test_reference_routing():
    routing = load_module("h3_reference_routing")
    bank = {
        "entries": [
            {
                "kind": "image",
                "items": [{"type": "image", "data": "p1"}],
                "blocks": ["pb1"],
                "picture_label": 1,
            },
            {
                "kind": "image",
                "items": [{"type": "image", "data": "p2"}],
                "blocks": ["pb2"],
                "picture_label": 2,
            },
            {
                "kind": "audio",
                "items": [{"type": "audio"}],
                "blocks": ["ab1"],
                "audio_label": 1,
            },
        ]
    }
    prompt, _items, blocks, report = routing.route_reference_bank(
        bank,
        (
            "<Picture 2> is the actor. <Audio 1> is their voice (S1). "
            "(S1) says: <d>Hi</d>"
        ),
        0,
        mode="auto_speaker_aware",
        visual_mode="auto_prompt_aware",
    )
    assert "<Picture 1>" in prompt
    assert "<Audio 1>" in prompt
    assert blocks == ["pb2", "ab1"]
    assert "pictures source=[2]" in report

    prompt, _items, blocks, _report = routing.route_reference_bank(
        bank,
        "<Picture 1> unused",
        0,
        mode="schedule",
        schedule="none",
        visual_mode="schedule",
        visual_schedule="P2|P1",
    )
    assert blocks == ["pb2"]
    assert "inactive picture reference" in prompt

    _prompt, _items, blocks, _report = routing.route_reference_bank(
        bank,
        "<Picture 1> omitted because the visual schedule ended",
        1,
        mode="schedule",
        schedule="none|none",
        visual_mode="schedule",
        visual_schedule="P1",
    )
    assert blocks == []


if __name__ == "__main__":
    test_frame_schedule()
    test_reference_routing()
    print("Variable multishot tests passed.")
