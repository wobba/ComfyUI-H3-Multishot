# -*- coding: utf-8 -*-
"""Per-shot routing for persistent MiniMax-H3 reference banks."""

import json
import re


_AUDIO_TAG = re.compile(r"<Audio\s+(\d+)>", re.IGNORECASE)
_PICTURE_TAG = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)
_VIDEO_TAG = re.compile(r"<Video\s+(\d+)>", re.IGNORECASE)
_AUDIO_DEFINITION = re.compile(
    r"<Audio\s+(\d+)>\s+is\b.*?\(S(\d+)\)", re.IGNORECASE | re.DOTALL
)


def _audio_definition_map(prompt):
    return {int(audio): int(speaker) for audio, speaker in _AUDIO_DEFINITION.findall(prompt)}


def _speaker_has_dialogue(prompt, speaker):
    pattern = re.compile(
        rf"\(S{speaker}\)(?:(?!\(S\d+\)).)*?"
        rf"(?:says|replies|asks|shouts|whispers|answers)\s*:\s*<d>",
        re.IGNORECASE | re.DOTALL,
    )
    return bool(pattern.search(prompt))


def _explicitly_inactive(prompt, audio_label):
    patterns = (
        rf"<Audio\s+{audio_label}>[^.\n]{{0,240}}\bnot\s+used\b",
        rf"\bnot\s+used\b[^.\n]{{0,240}}<Audio\s+{audio_label}>",
    )
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns)


def _scheduled_labels(schedule, shot_index):
    """Return source audio labels selected for the shot, or None without a schedule."""
    text = (schedule or "").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("shots", data)
        if not isinstance(data, list):
            raise ValueError("JSON schedule must be a list or {\"shots\": [...]} object")
        value = data[min(shot_index, len(data) - 1)] if data else []
    except json.JSONDecodeError:
        values = [item.strip() for item in re.split(r"[|;\n]", text) if item.strip()]
        value = values[min(shot_index, len(values) - 1)] if values else ""

    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        return {int(number) for number in re.findall(r"\d+", value)}
    if isinstance(value, list):
        return {int(number) for number in value}
    raise ValueError(f"Unsupported audio schedule entry for shot {shot_index + 1}: {value!r}")


def _auto_labels(prompt, available_labels):
    """Select audio refs only when their mapped speaker actually speaks.

    Unknown mappings are retained so ambiguous prose never silently drops a
    user-provided reference. Explicit "not used" text always wins.
    """
    speaker_map = _audio_definition_map(prompt)
    active = set()
    for label in available_labels:
        if _explicitly_inactive(prompt, label):
            continue
        speaker = speaker_map.get(label)
        if speaker is None:
            # A shot that does not name this reference has no reason to pay
            # for its packed audio rows. A bare tag remains an explicit use.
            if re.search(rf"<Audio\s+{label}>", prompt, re.IGNORECASE):
                active.add(label)
        elif _speaker_has_dialogue(prompt, speaker):
            active.add(label)
    return active


def _replace_audio_labels(prompt, source_to_local, inactive):
    def replace(match):
        source = int(match.group(1))
        if source in source_to_local:
            return f"<Audio {source_to_local[source]}>"
        if source in inactive:
            return "the inactive reference audio"
        return match.group(0)

    return _AUDIO_TAG.sub(replace, prompt)


def _scheduled_visual_labels(schedule, shot_index):
    """Return ({pictures}, {videos}) selected for a segment."""
    text = (schedule or "").strip()
    if not text:
        return set(), set()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("shots", data.get("segments", data))
        if not isinstance(data, list):
            raise ValueError(
                "Visual reference schedule must be a list or "
                "{\"segments\": [...]} object"
            )
        value = data[shot_index] if shot_index < len(data) else ""
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            value = ",".join(
                [f"P{item}" for item in value.get("pictures", [])]
                + [f"V{item}" for item in value.get("videos", [])]
            )
        else:
            value = str(value or "")
    except json.JSONDecodeError:
        values = re.split(r"[|;\n]", text)
        value = values[shot_index] if shot_index < len(values) else ""

    pictures = {
        int(label)
        for label in re.findall(r"(?:P|Picture)\s*(\d+)", value, re.IGNORECASE)
    }
    videos = {
        int(label)
        for label in re.findall(r"(?:V|Video)\s*(\d+)", value, re.IGNORECASE)
    }
    return pictures, videos


def _auto_visual_labels(prompt, available_pictures, available_videos):
    pictures = {
        int(label) for label in _PICTURE_TAG.findall(prompt)
        if int(label) in available_pictures
    }
    videos = {
        int(label) for label in _VIDEO_TAG.findall(prompt)
        if int(label) in available_videos
    }
    return pictures, videos


def _replace_visual_labels(prompt, picture_map, video_map,
                           inactive_pictures, inactive_videos):
    def replace_picture(match):
        source = int(match.group(1))
        if source in picture_map:
            return f"<Picture {picture_map[source]}>"
        if source in inactive_pictures:
            return "the inactive picture reference"
        return match.group(0)

    def replace_video(match):
        source = int(match.group(1))
        if source in video_map:
            return f"<Video {video_map[source]}>"
        if source in inactive_videos:
            return "the inactive video reference"
        return match.group(0)

    return _VIDEO_TAG.sub(
        replace_video,
        _PICTURE_TAG.sub(replace_picture, prompt),
    )


def route_reference_bank(
    bank,
    prompt,
    shot_index,
    mode="always",
    schedule="",
    visual_mode="always",
    visual_schedule="",
):
    """Return prompt/items/blocks for one segment with only active refs."""
    if not bank or not bank.get("entries"):
        return prompt, list((bank or {}).get("items", [])), list((bank or {}).get("blocks", [])), ""

    entries = bank["entries"]
    available_pictures = {
        entry["picture_label"]
        for entry in entries
        if entry.get("picture_label") is not None
    }
    available_videos = {
        entry["video_label"]
        for entry in entries
        if entry.get("video_label") is not None
    }
    if visual_mode == "always":
        active_pictures = available_pictures
        active_videos = available_videos
    elif visual_mode == "schedule":
        active_pictures, active_videos = _scheduled_visual_labels(
            visual_schedule, shot_index
        )
    else:
        active_pictures, active_videos = _auto_visual_labels(
            prompt, available_pictures, available_videos
        )

    unknown_pictures = active_pictures - available_pictures
    unknown_videos = active_videos - available_videos
    if unknown_pictures or unknown_videos:
        raise ValueError(
            f"Segment {shot_index + 1} selects unavailable visual references: "
            f"pictures={sorted(unknown_pictures)}, videos={sorted(unknown_videos)}"
        )

    available_audio = {
        entry["audio_label"]
        for entry in entries
        if entry.get("audio_label") is not None
    }

    if mode == "always":
        active_audio = available_audio
    else:
        scheduled = _scheduled_labels(schedule, shot_index)
        active_audio = scheduled if scheduled is not None else _auto_labels(prompt, available_audio)
        unknown = active_audio - available_audio
        if unknown:
            raise ValueError(
                f"Shot {shot_index + 1} selects unavailable audio reference(s): {sorted(unknown)}"
            )

    items = []
    blocks = []
    audio_map = {}
    picture_map = {}
    video_map = {}
    local_audio_label = 0
    local_picture_label = 0
    local_video_label = 0
    for entry in entries:
        picture_label = entry.get("picture_label")
        video_label = entry.get("video_label")
        if picture_label is not None and picture_label not in active_pictures:
            continue
        if video_label is not None and video_label not in active_videos:
            continue

        audio_label = entry.get("audio_label")
        include_audio = audio_label is None or audio_label in active_audio
        if include_audio:
            items.extend(entry["items"])
            blocks.extend(entry["blocks"])
            if audio_label is not None:
                local_audio_label += 1
                audio_map[audio_label] = local_audio_label
        elif entry["kind"] == "video":
            # Keep a video reference's visual context while dropping only its
            # paired soundtrack. Its audio label does not consume context rows.
            items.append(entry["video_item"])
            block = dict(entry["blocks"][0])
            block["kind"] = "video"
            block["ref_audio_t"] = 0
            block["audio_latent"] = None
            blocks.append(block)

        if picture_label is not None:
            local_picture_label += 1
            picture_map[picture_label] = local_picture_label
        if video_label is not None:
            local_video_label += 1
            video_map[video_label] = local_video_label

    inactive_audio = available_audio - set(audio_map)
    inactive_pictures = available_pictures - active_pictures
    inactive_videos = available_videos - active_videos
    rewritten_prompt = _replace_visual_labels(
        _replace_audio_labels(prompt, audio_map, inactive_audio),
        picture_map,
        video_map,
        inactive_pictures,
        inactive_videos,
    )
    report = (
        f"pictures source={sorted(active_pictures)} -> local={picture_map}; "
        f"videos source={sorted(active_videos)} -> local={video_map}; "
        f"audio source={sorted(set(audio_map))} -> local={audio_map}; "
        f"skipped audio={sorted(inactive_audio)}"
    )
    return rewritten_prompt, items, blocks, report
