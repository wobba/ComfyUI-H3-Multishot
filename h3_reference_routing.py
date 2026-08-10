# -*- coding: utf-8 -*-
"""Per-shot routing for persistent MiniMax-H3 reference banks."""

import json
import re


_AUDIO_TAG = re.compile(r"<Audio\s+(\d+)>", re.IGNORECASE)
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
            # A bare tag in the action prose is an explicit request. A tag
            # limited to a definition is ambiguous, so preserve it safely.
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


def route_reference_bank(bank, prompt, shot_index, mode="always", schedule=""):
    """Return prompt/items/blocks for one shot with only active audio refs."""
    if not bank or not bank.get("entries"):
        return prompt, list((bank or {}).get("items", [])), list((bank or {}).get("blocks", [])), ""

    entries = bank["entries"]
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
    source_to_local = {}
    local_audio_label = 0
    for entry in entries:
        audio_label = entry.get("audio_label")
        include_audio = audio_label is None or audio_label in active_audio
        if include_audio:
            items.extend(entry["items"])
            blocks.extend(entry["blocks"])
            if audio_label is not None:
                local_audio_label += 1
                source_to_local[audio_label] = local_audio_label
        elif entry["kind"] == "video":
            # Keep a video reference's visual context while dropping only its
            # paired soundtrack. Its audio label does not consume context rows.
            items.append(entry["video_item"])
            block = dict(entry["blocks"][0])
            block["kind"] = "video"
            block["ref_audio_t"] = 0
            block["audio_latent"] = None
            blocks.append(block)

    inactive = available_audio - active_audio
    rewritten_prompt = _replace_audio_labels(prompt, source_to_local, inactive)
    report = (
        f"audio refs source={sorted(active_audio)} -> local={source_to_local}; "
        f"skipped={sorted(inactive)}"
    )
    return rewritten_prompt, items, blocks, report
