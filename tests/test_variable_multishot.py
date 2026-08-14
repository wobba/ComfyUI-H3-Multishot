import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inline_frame_counts():
    utils = load_module("h3_multishot_utils")
    prompts, frames = utils._resolve_segment_frames(
        [
            "frame_count: 124\nFirst prompt",
            "Second prompt",
            "frame_count: 362\nThird prompt",
        ],
        243,
    )
    assert frames == [124, 243, 362]
    assert prompts == ["First prompt", "Second prompt", "Third prompt"]
    prompts, frames = utils._resolve_segment_frames(
        ["frame_count: 124\nFirst prompt", "Second prompt"],
        175,
    )
    assert frames == [124, 175]
    assert utils._resolve_script_input("fallback", None) == "fallback"
    assert utils._resolve_script_input("fallback", "  ") == "fallback"
    assert utils._resolve_script_input("fallback", "external") == "external"


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


def test_disk_manifest_helpers():
    disk = load_module("h3_disk_sampler")
    assert disk._validate_run_name("movie-01_take.2") == "movie-01_take.2"
    try:
        disk._validate_run_name("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe run_name was accepted")
    assert disk._plan_hash({"b": 2, "a": 1}) == disk._plan_hash(
        {"a": 1, "b": 2}
    )
    generated, was_generated = disk._resolve_run_name("")
    assert was_generated and generated.startswith("h3_")
    assert disk._legacy_settings_match(
        {"seed": 1, "plan_tag": "old-manual-tag"},
        {
            "seed": 1,
            "plan_tag": "",
            "primary_loras": ["auto"],
            "gpu_cleanup_between_segments": True,
        },
    )
    import torch
    assert disk._tensor_fingerprint(torch.zeros(1, 8)) != (
        disk._tensor_fingerprint(torch.ones(1, 8))
    )

    bank = load_module("h3_reference_bank")
    import hashlib
    first = hashlib.sha256()
    second = hashlib.sha256()
    bank._update_tensor_fingerprint(first, "image:1", torch.zeros(1, 4, 4, 3))
    bank._update_tensor_fingerprint(second, "image:1", torch.ones(1, 4, 4, 3))
    assert first.hexdigest() != second.hexdigest()


def test_lazy_model_route():
    router = load_module("h3_model_router").H3ModelRoute()
    fl2va = object()
    ref2va = object()
    assert router.check_lazy_status("FL2VA only", None, None) == [
        "fl2va_model"
    ]
    assert router.check_lazy_status("Ref2VA only", None, None) == [
        "ref2va_model"
    ]
    assert router.check_lazy_status("Mixed per segment", fl2va, None) == [
        "ref2va_model"
    ]
    assert router.route("FL2VA only", fl2va, None) == (fl2va, None)
    assert router.route("Ref2VA only", None, ref2va) == (ref2va, None)
    assert router.route("Mixed per segment", fl2va, ref2va) == (
        fl2va, ref2va
    )


if __name__ == "__main__":
    test_inline_frame_counts()
    test_reference_routing()
    test_disk_manifest_helpers()
    test_lazy_model_route()
    print("Variable multishot tests passed.")
