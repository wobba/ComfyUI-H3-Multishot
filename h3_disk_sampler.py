# -*- coding: utf-8 -*-
"""Disk-backed, resumable long-form MiniMax H3 Memory sampler."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import wave


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _validate_run_name(value):
    value = (value or "").strip()
    if not _SAFE_RUN_NAME.fullmatch(value):
        raise ValueError(
            "run_name must start with a letter or number and contain only "
            "letters, numbers, dot, underscore, or hyphen (maximum 96 chars)"
        )
    return value


def _resolve_run_name(value):
    value = (value or "").strip()
    if value:
        return _validate_run_name(value), False
    generated = (
        "h3_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    return generated, True


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _plan_hash(settings):
    payload = json.dumps(
        settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_settings_match(previous, current):
    for key, value in previous.items():
        if key == "plan_tag" and not current.get("plan_tag"):
            continue
        if current.get(key) != value:
            return False
    return True


def _changed_setting_keys(previous, current):
    return sorted(
        key
        for key in set(previous) | set(current)
        if previous.get(key) != current.get(key)
    )


def _restart_manifest_from_segment(
    root,
    manifest,
    segment_number,
    settings,
    plan_hash,
):
    """Discard a rendered suffix while retaining its durable prefix."""
    segment_number = int(segment_number)
    if segment_number < 1:
        raise ValueError("restart_from_segment must be 0 (off) or at least 1")

    entries = list(manifest.get("segments", []))
    restart_index = segment_number - 1
    if restart_index > len(entries):
        raise RuntimeError(
            f"Cannot restart from segment {segment_number}; only "
            f"{len(entries)} segment(s) are durable."
        )

    new_frames = list(settings.get("segment_frames", []))
    if restart_index > len(new_frames):
        raise RuntimeError(
            f"Cannot preserve {restart_index} segment(s) when the new plan "
            f"contains only {len(new_frames)}."
        )
    for index, entry in enumerate(entries[:restart_index]):
        if entry.get("frame_count") != new_frames[index]:
            raise RuntimeError(
                f"Cannot preserve segment {index + 1}: its frame count changed "
                f"from {entry.get('frame_count')} to {new_frames[index]}."
            )

    previous_settings = manifest.get("settings", {})
    if (
        previous_settings.get("width") != settings.get("width")
        or previous_settings.get("height") != settings.get("height")
    ):
        raise RuntimeError(
            "Cannot preserve rendered segments after changing width or height."
        )

    for entry in entries[restart_index:]:
        for key in ("file", "last_frame"):
            relative = entry.get(key)
            if relative:
                path = root / relative
                if path.is_file():
                    path.unlink()

    final_relative = manifest.get("final_video")
    if final_relative:
        final_path = root / final_relative
        if final_path.is_file():
            final_path.unlink()

    if restart_index == 0:
        anchor_relative = manifest.get("anchor_frame")
        if anchor_relative:
            anchor_path = root / anchor_relative
            if anchor_path.is_file():
                anchor_path.unlink()
        manifest["anchor_frame"] = None

    changed = _changed_setting_keys(previous_settings, settings)
    manifest.setdefault("restarts", []).append({
        "from_segment": segment_number,
        "previous_plan_hash": manifest.get("plan_hash"),
        "changed_settings": changed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    manifest["segments"] = entries[:restart_index]
    manifest["settings"] = settings
    manifest["plan_hash"] = plan_hash
    manifest["status"] = "rendering"
    manifest["final_video"] = None
    return changed


def _tensor_fingerprint(tensor, sample_count=4096):
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    flat = value.reshape(-1)
    if flat.numel() > sample_count:
        indices = torch.linspace(
            0, flat.numel() - 1, sample_count, dtype=torch.int64
        )
        flat = flat[indices]
    if flat.is_floating_point():
        flat = flat.to(torch.float32)
    digest.update(flat.numpy().tobytes())
    return digest.hexdigest()


def _model_source(model):
    if model is None or not hasattr(model, "get_attachment"):
        return None
    return model.get_attachment("h3_source_model_name")


def _model_lora_tags(model):
    if model is None or not hasattr(model, "get_attachment"):
        return []
    return list(model.get_attachment("h3_lora_plan_tags") or ())


def _clip_source(clip):
    patcher = getattr(clip, "patcher", None)
    if patcher is None or not hasattr(patcher, "get_attachment"):
        return None
    return patcher.get_attachment("h3_source_clip_name")


def _image_to_uint8(image):
    import torch

    frame = image[0] if image.ndim == 4 else image
    return (
        frame[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )


def _save_frame(image, path):
    from PIL import Image

    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(_image_to_uint8(image), mode="RGB").save(
        temporary, format="PNG"
    )
    os.replace(temporary, path)


def _load_frame(path):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _write_audio_wav(audio, sample_rate, path):
    import torch

    waveform = audio.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.clamp(-1, 1)
    pcm = (
        waveform.transpose(0, 1)
        .mul(32767)
        .round()
        .to(torch.int16)
        .contiguous()
        .numpy()
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(pcm.shape[1]))
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm.tobytes())


def _encode_segment(images, waveform, sample_rate, output_path, fps=24):
    """Encode one decoded segment to a lossless MKV without a full uint8 copy."""
    import torch

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for H3 disk-backed sampling")

    images = images.detach().to(device="cpu", dtype=torch.float32)
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(
            f"Expected IMAGE [frames,height,width,channels], got {images.shape}"
        )
    frame_count, height, width = images.shape[:3]
    if frame_count < 1:
        raise ValueError("Cannot encode an empty H3 segment")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".partial.mkv")
    audio_path = output_path.with_suffix(".audio.wav")
    partial.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    _write_audio_wav(waveform, sample_rate, audio_path)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-g",
        "1",
        "-pix_fmt",
        "bgr0",
        "-c:a",
        "pcm_s16le",
        str(partial),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for start in range(0, frame_count, 4):
            chunk = (
                images[start:start + 4, ..., :3]
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .contiguous()
                .numpy()
            )
            process.stdin.write(chunk.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"ffmpeg segment encode failed ({return_code}): "
                f"{error[-4000:]}"
            )
        os.replace(partial, output_path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        partial.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def _concat_segments(segment_paths, output_path):
    """Assemble video with 40ms equal-power audio crossfades."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for H3 disk-backed sampling")
    if not segment_paths:
        raise ValueError("No completed H3 segments to concatenate")

    partial = output_path.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)
    try:
        common = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        for path in segment_paths:
            common.extend(["-i", str(path)])

        if len(segment_paths) == 1:
            filters = []
            maps = ["-map", "0:v:0", "-map", "0:a:0"]
        else:
            filters = []
            video_inputs = []
            audio_labels = []
            for index in range(len(segment_paths)):
                filters.append(
                    f"[{index}:v]setpts=PTS-STARTPTS[v{index}]"
                )
                filters.append(
                    f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]"
                )
                video_inputs.append(f"[v{index}]")
                audio_labels.append(f"a{index}")
            filters.append(
                "".join(video_inputs)
                + f"concat=n={len(segment_paths)}:v=1:a=0[vout]"
            )
            current_audio = audio_labels[0]
            for index in range(1, len(audio_labels)):
                output_audio = f"ax{index}"
                filters.append(
                    f"[{current_audio}][{audio_labels[index]}]"
                    f"acrossfade=d={1.0 / 24.0:.10f}:c1=qsin:c2=qsin"
                    f"[{output_audio}]"
                )
                current_audio = output_audio
            maps = ["-map", "[vout]", "-map", f"[{current_audio}]"]
            common.extend(["-filter_complex", ";".join(filters)])

        nvenc = common + [
            *maps,
            "-c:v", "h264_nvenc",
            "-preset", "p7",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "14",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "320k",
            "-movflags", "+faststart",
            str(partial),
        ]
        result = subprocess.run(
            nvenc,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            partial.unlink(missing_ok=True)
            fallback = common + [
                *maps,
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "14",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "320k",
                "-movflags", "+faststart",
                str(partial),
            ]
            result = subprocess.run(
                fallback,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"ffmpeg final assembly failed ({result.returncode}): "
                    f"{result.stderr[-4000:]}"
                )
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)


def _relative(path, root):
    return str(path.relative_to(root)).replace("\\", "/")


def _ui_video_result(final_path, result):
    """Present the finished master in the node itself.

    Without this the node saves a perfectly good MP4 that never appears in the
    UI, so workflows bolt a SaveVideo onto the VIDEO output purely for the
    preview - which writes a second copy of the same video under a different
    prefix. Emitting the same payload SaveVideo does removes that duplicate.
    """
    try:
        import folder_paths
        output_root = Path(folder_paths.get_output_directory())
        subfolder = str(final_path.parent.relative_to(output_root)).replace("\\", "/")
    except (ImportError, ValueError):
        # Rendered outside the output tree: no preview, but still return video.
        return result
    return {
        "ui": {
            "images": [{
                "filename": final_path.name,
                "subfolder": subfolder,
                "type": "output",
            }],
            "animated": (True,),
        },
        "result": result,
    }


def _release_process_memory():
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    rss_gib = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_gib = int(line.split()[1]) / (1024 ** 2)
                break
    except Exception:
        pass
    if rss_gib is not None:
        print(f"[H3Disk] process RSS after cleanup: {rss_gib:.2f} GiB",
              flush=True)


class H3MultishotMemoryDiskSampler:
    """Memory chaining with immediate segment persistence and manifest resume."""

    @classmethod
    def INPUT_TYPES(cls):
        from .h3_multishot_utils import H3MultishotMemorySampler

        schema = copy.deepcopy(H3MultishotMemorySampler.INPUT_TYPES())
        schema["required"]["run_name"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": "Leave empty for a unique logged name per queue. "
                           "Paste a prior generated name here to resume it.",
            },
        )
        schema["optional"]["resume"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Resume completed segments when the manifest plan "
                           "matches this run.",
            },
        )
        schema["optional"]["keep_segments"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Keep lossless segment MKVs after final assembly. "
                           "PNG checkpoints and manifest are always retained.",
            },
        )
        schema["optional"]["plan_tag"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": "Optional manual revision note. Tracked LoRA nodes "
                           "automatically fingerprint LoRA name and strength.",
            },
        )
        schema["optional"]["gpu_cleanup_between_segments"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Fully unload GPU models and clear CUDA allocator "
                           "state after each durable segment. Recommended for "
                           "long full-INT8 runs.",
            },
        )
        schema["optional"]["restart_from_segment"] = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 64,
                "tooltip": "0 = normal resume. Set a 1-based segment number "
                           "to delete that durable segment and everything after "
                           "it, then rerender the suffix with current settings.",
            },
        )
        return schema

    RETURN_TYPES = ("VIDEO", "STRING", "INT")
    RETURN_NAMES = ("video", "manifest_path", "segments_rendered")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"
    OUTPUT_NODE = True

    def run(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        script,
        shot_count,
        width,
        height,
        frames_per_shot,
        seed,
        steps,
        seed_per_shot,
        memory_frames,
        anchor_frames,
        run_name,
        start_image=None,
        sampler_name="res_multistep",
        scheduler="simple",
        persistent_refs=None,
        ref2va_model=None,
        audio_reference_mode="always",
        audio_reference_schedule="",
        visual_reference_mode="always",
        visual_reference_schedule="",
        script_override=None,
        resume=True,
        keep_segments=False,
        plan_tag="",
        gpu_cleanup_between_segments=True,
        restart_from_segment=0,
    ):
        import gc
        import folder_paths
        from comfy_api.latest import InputImpl
        from comfy_extras import nodes_minimax_h3 as mmh3
        import comfy.model_management as model_management
        from .h3_multishot_utils import (
            _iter_memory_segments,
            _prepare_memory_plan,
        )

        from .h3_multishot_utils import _resolve_script_input

        script = _resolve_script_input(script, script_override)
        run_name, generated_run_name = _resolve_run_name(run_name)
        if generated_run_name:
            print(f"[H3Disk] auto run_name: {run_name}", flush=True)
        (
            shots,
            segment_count,
            segment_frames,
            segment_seam_blends,
        ) = _prepare_memory_plan(
            script,
            shot_count,
            frames_per_shot,
            mmh3.align_frame_count,
        )
        settings = {
            "script": script,
            "shot_count": shot_count,
            "width": width,
            "height": height,
            "frames_per_shot": frames_per_shot,
            "segment_frames": segment_frames,
            "segment_seam_blends": segment_seam_blends,
            "seed": seed,
            "steps": steps,
            "seed_per_shot": seed_per_shot,
            "memory_frames": memory_frames,
            "anchor_frames": anchor_frames,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "audio_reference_mode": audio_reference_mode,
            "audio_reference_schedule": audio_reference_schedule,
            "visual_reference_mode": visual_reference_mode,
            "visual_reference_schedule": visual_reference_schedule,
            "reference_report": (persistent_refs or {}).get("report", ""),
            "reference_fingerprint": (
                persistent_refs or {}
            ).get("content_fingerprint", ""),
            "start_image_fingerprint": (
                _tensor_fingerprint(start_image)
                if start_image is not None else ""
            ),
            "primary_model": _model_source(model),
            "ref2va_model": _model_source(ref2va_model),
            "primary_loras": _model_lora_tags(model),
            "ref2va_loras": _model_lora_tags(ref2va_model),
            "text_encoder": _clip_source(clip),
            "plan_tag": str(plan_tag or ""),
            "gpu_cleanup_between_segments": bool(
                gpu_cleanup_between_segments
            ),
        }
        plan_hash = _plan_hash(settings)
        root = (
            Path(folder_paths.get_output_directory())
            / "H3_DISK"
            / run_name
        )
        manifest_path = root / "manifest.json"
        final_path = root / "master.mp4"
        root.mkdir(parents=True, exist_ok=True)

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not resume:
                raise RuntimeError(
                    f"Run {run_name!r} already exists. Enable resume or choose "
                    "a new run_name."
                )
            if restart_from_segment:
                changed = _restart_manifest_from_segment(
                    root,
                    manifest,
                    restart_from_segment,
                    settings,
                    plan_hash,
                )
                _atomic_json(manifest_path, manifest)
                print(
                    f"[H3Disk] restarting {run_name} from segment "
                    f"{restart_from_segment}; changed settings: "
                    f"{', '.join(changed) if changed else 'none'}",
                    flush=True,
                )
            elif manifest.get("plan_hash") != plan_hash:
                previous_settings = manifest.get("settings", {})
                legacy_match = _legacy_settings_match(
                    previous_settings, settings
                )
                if legacy_match:
                    manifest["settings"] = settings
                    manifest["plan_hash"] = plan_hash
                    _atomic_json(manifest_path, manifest)
                    print(
                        f"[H3Disk] upgraded legacy resume fingerprint for "
                        f"{run_name}",
                        flush=True,
                    )
                else:
                    changed = _changed_setting_keys(
                        previous_settings, settings
                    )
                    raise RuntimeError(
                        f"Run {run_name!r} exists with different settings. "
                        f"Changed: {', '.join(changed)}. Choose a new run_name "
                        "or set restart_from_segment to rerender a suffix."
                    )
        else:
            existing = [path for path in root.iterdir()]
            if existing:
                raise RuntimeError(
                    f"Run folder is non-empty without a manifest: {root}"
                )
            manifest = {
                "version": 1,
                "status": "rendering",
                "plan_hash": plan_hash,
                "settings": settings,
                "anchor_frame": None,
                "segments": [],
                "final_video": None,
            }
            _atomic_json(manifest_path, manifest)

        if (
            manifest.get("status") == "complete"
            and manifest.get("final_video")
        ):
            completed_final = root / manifest["final_video"]
            if completed_final.is_file():
                print(f"[H3Disk] completed run reused: {completed_final}",
                      flush=True)
                return _ui_video_result(completed_final, (
                    InputImpl.VideoFromFile(str(completed_final)),
                    str(manifest_path),
                    segment_count,
                ))

        entries = manifest.get("segments", [])
        for expected, entry in enumerate(entries):
            if entry.get("index") != expected:
                raise RuntimeError(
                    "Manifest segments are not a contiguous zero-based prefix"
                )
            if not (root / entry["file"]).is_file():
                raise RuntimeError(
                    f"Manifest segment file is missing: {entry['file']}"
                )
        start_index = len(entries)
        if start_index > segment_count:
            raise RuntimeError("Manifest has more segments than the render plan")

        anchor = (
            start_image[:1]
            if start_image is not None and start_index == 0
            else None
        )
        anchor_relative = manifest.get("anchor_frame")
        if start_index and anchor_relative:
            anchor = _load_frame(root / anchor_relative)
        history = [
            _load_frame(root / entry["last_frame"])
            for entry in entries[-8:]
        ]

        if start_index:
            print(f"[H3Disk] resuming {run_name} at segment "
                  f"{start_index + 1}/{segment_count}", flush=True)

        generator = _iter_memory_segments(
            model=model,
            clip=clip,
            video_vae=video_vae,
            audio_vae=audio_vae,
            shots=shots,
            segment_frames=segment_frames,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            memory_frames=memory_frames,
            anchor_frames=anchor_frames,
            seed_per_shot=seed_per_shot,
            sampler_name=sampler_name,
            scheduler=scheduler,
            persistent_refs=persistent_refs,
            ref2va_model=ref2va_model,
            visual_reference_mode=visual_reference_mode,
            visual_reference_schedule=visual_reference_schedule,
            audio_reference_mode=audio_reference_mode,
            audio_reference_schedule=audio_reference_schedule,
            segment_seam_blends=segment_seam_blends,
            start_index=start_index,
            anchor=anchor,
            history=history,
            announce_start_image=start_image is not None and start_index == 0,
            trim_audio_seam=False,
        )

        for segment in generator:
            index = segment["index"]
            segment_path = root / f"segment_{index + 1:04d}.mkv"
            last_frame_path = root / f"segment_{index + 1:04d}_last.png"
            print(f"[H3Disk] encoding segment {index + 1}/{segment_count} "
                  f"to {segment_path}", flush=True)
            _encode_segment(
                segment["images"],
                segment["waveform"],
                segment["sample_rate"],
                segment_path,
            )
            _save_frame(segment["history"][-1], last_frame_path)

            if (
                anchor_frames > 0
                and manifest.get("anchor_frame") is None
                and segment["anchor"] is not None
            ):
                anchor_path = root / "anchor.png"
                _save_frame(segment["anchor"], anchor_path)
                manifest["anchor_frame"] = _relative(anchor_path, root)

            manifest["segments"].append({
                "index": index,
                "file": _relative(segment_path, root),
                "last_frame": _relative(last_frame_path, root),
                "frame_count": segment_frames[index],
                "seam_blend_frames": segment_seam_blends[index],
                "output_frame_count": int(segment["images"].shape[0]),
                "sample_rate": int(segment["sample_rate"]),
                "model_mode": segment["model_mode"],
                "route_report": segment["route_report"],
                "segment_seed": (
                    seed + index if seed_per_shot else seed
                ),
            })
            _atomic_json(manifest_path, manifest)
            # The generator holds the yielded dictionary while suspended.
            # Clearing it releases decoded video/audio before model CPU offload.
            segment.clear()
            del segment
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass
            _release_process_memory()
            if gpu_cleanup_between_segments:
                print("[H3Disk] unloading GPU models before next segment",
                      flush=True)
                model_management.unload_all_models()
                model_management.cleanup_models()
                model_management.soft_empty_cache(force=True)
                _release_process_memory()

        segment_paths = [
            root / entry["file"] for entry in manifest["segments"]
        ]
        _concat_segments(segment_paths, final_path)
        manifest["status"] = "complete"
        manifest["final_video"] = _relative(final_path, root)
        _atomic_json(manifest_path, manifest)

        if not keep_segments:
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)

        print(f"[H3Disk] complete: {final_path}", flush=True)
        return _ui_video_result(final_path, (
            InputImpl.VideoFromFile(str(final_path)),
            str(manifest_path),
            segment_count,
        ))


NODE_CLASS_MAPPINGS = {
    "H3MultishotMemoryDiskSampler": H3MultishotMemoryDiskSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MultishotMemoryDiskSampler":
        "H3 Multishot Memory Sampler (Disk / Resume)",
}
