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
    import torch
    audio = utils._xfade_audio(
        [torch.zeros(1, 1, 3200), torch.zeros(1, 1, 3200)],
        32000,
    )
    assert audio.shape[-1] == 6400 - int(32000 / 24)
    prompts, blends = utils._extract_inline_seam_blends([
        "First",
        "seam_blend_frames: 6\nSecond",
    ])
    assert prompts == ["First", "Second"]
    assert blends == [0, 6]


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


def test_dialogue_driven_audio_routing():
    routing = load_module("h3_reference_routing")
    bank = {
        "entries": [
            {
                "kind": "audio",
                "items": [{"type": "audio", "data": "a1"}],
                "blocks": ["ab1"],
                "audio_label": 1,
            },
            {
                "kind": "audio",
                "items": [{"type": "audio", "data": "a2"}],
                "blocks": ["ab2"],
                "audio_label": 2,
            },
        ]
    }
    declarations = (
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).\n"
        "<Audio 2> is the voice-timbre reference for <Subject 2> (S2).\n"
    )

    # A silent segment pays for no voices at all, even though both are declared.
    _prompt, items, blocks, _report = routing.route_reference_bank(
        bank,
        declarations + "[Shot 1] The two walk through the lobby without speaking.",
        0,
        mode="auto_speaker_aware",
    )
    assert (items, blocks) == ([], [])

    # Only speaker 2 speaks, so source <Audio 2> is sent and renumbered local 1.
    prompt, items, blocks, report = routing.route_reference_bank(
        bank,
        declarations + "[Shot 1] <Subject 2> (S2) leans in and pitches:\n<d>Hei.</d>",
        0,
        mode="auto_speaker_aware",
    )
    assert blocks == ["ab2"]
    assert items == [{"type": "audio", "data": "a2"}]
    assert "<Audio 1> is the voice-timbre reference for <Subject 2>" in prompt
    assert "<Audio 2>" not in prompt
    assert "audio source=[2] -> local={2: 1}" in report

    # A bank can hold Audio 1 for another segment. It has no declaration here,
    # so S2's line must not keep it alive merely because it exists in the bank.
    prompt, _items, blocks, report = routing.route_reference_bank(
        bank,
        "<Audio 2>: reference - timbre guides <Subject 2> (S2).\n"
        "[Shot 1] <Subject 2> (S2) smiles and says:\n<d>Hei.</d>",
        0,
        mode="auto_speaker_aware",
    )
    assert blocks == ["ab2"]
    assert "<Audio 1>: reference - timbre guides <Subject 2>" in prompt
    assert "audio source=[2] -> local={2: 1}" in report
    assert "skipped audio=[1]" in report

    # Dialogue with no speaker tag in reach keeps every voice eligible.
    _prompt, _items, blocks, _report = routing.route_reference_bank(
        bank,
        declarations + "[Shot 1] A voice answers from the dark.\n<d>Hei.</d>",
        0,
        mode="auto_speaker_aware",
    )
    assert blocks == ["ab1", "ab2"]


def test_subject_aware_visual_routing():
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
        ]
    }
    declarations = (
        "<Subject 1> (S1) is Andreas, the cyclist in <Picture 1>.\n"
        "<Picture 1> is the identity reference for <Subject 1>.\n"
        "<Subject 2> (S2) is Maria, the reporter in <Picture 2>.\n"
        "<Picture 2> is the identity reference for <Subject 2>.\n"
    )

    # Maria is declared but absent, so her identity image is not packed.
    prompt, _items, blocks, report = routing.route_reference_bank(
        bank,
        declarations + "[Shot 1] <Subject 1> crests the climb alone.",
        0,
        visual_mode="auto_prompt_aware",
    )
    assert blocks == ["pb1"]
    assert "pictures source=[1] -> local={1: 1}" in report
    assert "the inactive picture reference" in prompt

    # A body that names no subject is not subject-styled: nothing is pruned.
    _prompt, _items, blocks, _report = routing.route_reference_bank(
        bank,
        declarations + "[Shot 1] Both riders crest the climb in <Picture 1> and <Picture 2>.",
        0,
        visual_mode="auto_prompt_aware",
    )
    assert blocks == ["pb1", "pb2"]


def test_disk_video_ui_result():
    disk = load_module("h3_disk_sampler")
    import sys
    import types
    from pathlib import Path

    stub = types.ModuleType("folder_paths")
    stub.get_output_directory = lambda: str(Path("/comfy/output"))
    sys.modules["folder_paths"] = stub
    try:
        final = Path("/comfy/output") / "H3_DISK" / "my-run" / "master.mp4"
        payload = disk._ui_video_result(final, ("video", "manifest", 3))

        # The node presents its own master, which is what removes the need for
        # a SaveVideo node writing a duplicate under output/video/.
        assert payload["result"] == ("video", "manifest", 3)
        assert payload["ui"]["images"] == [{
            "filename": "master.mp4",
            "subfolder": "H3_DISK/my-run",
            "type": "output",
        }]
        assert payload["ui"]["animated"] == (True,)

        # A path outside the output tree cannot be previewed; return unchanged.
        outside = Path("/somewhere/else/master.mp4")
        assert disk._ui_video_result(outside, ("video", "m", 1)) == ("video", "m", 1)
    finally:
        del sys.modules["folder_paths"]

    # Without ComfyUI present the helper must degrade, not raise.
    assert disk._ui_video_result(Path("/x/master.mp4"), ("v",)) == ("v",)


def test_disk_manifest_helpers():
    disk = load_module("h3_disk_sampler")
    import torch
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
    assert disk._changed_setting_keys(
        {"seed": 1, "width": 960},
        {"seed": 2, "width": 960},
    ) == ["seed"]
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in (
            "segment_0001.mkv",
            "segment_0001_last.png",
            "segment_0002.mkv",
            "segment_0002_last.png",
            "master.mp4",
            "anchor.png",
        ):
            (root / name).write_bytes(b"test")
        manifest = {
            "plan_hash": "old",
            "settings": {
                "seed": 1,
                "width": 960,
                "height": 544,
                "segment_frames": [243, 243],
            },
            "status": "complete",
            "anchor_frame": "anchor.png",
            "final_video": "master.mp4",
            "segments": [
                {
                    "index": 0,
                    "file": "segment_0001.mkv",
                    "last_frame": "segment_0001_last.png",
                    "frame_count": 243,
                },
                {
                    "index": 1,
                    "file": "segment_0002.mkv",
                    "last_frame": "segment_0002_last.png",
                    "frame_count": 243,
                },
            ],
        }
        changed = disk._restart_manifest_from_segment(
            root,
            manifest,
            2,
            {
                "seed": 2,
                "width": 960,
                "height": 544,
                "segment_frames": [243, 243],
            },
            "new",
        )
        assert changed == ["seed"]
        assert len(manifest["segments"]) == 1
        assert manifest["anchor_frame"] == "anchor.png"
        assert (root / "segment_0001.mkv").is_file()
        assert not (root / "segment_0002.mkv").exists()
        assert not (root / "master.mp4").exists()
        assert manifest["status"] == "rendering"
        assert manifest["plan_hash"] == "new"
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
    test_dialogue_driven_audio_routing()
    test_subject_aware_visual_routing()
    test_disk_manifest_helpers()
    test_disk_video_ui_result()
    test_lazy_model_route()
    print("Variable multishot tests passed.")
