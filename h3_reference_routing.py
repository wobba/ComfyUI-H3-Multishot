# -*- coding: utf-8 -*-
"""Per-shot routing for persistent MiniMax-H3 reference banks."""

import json
import re


_AUDIO_TAG = re.compile(r"<Audio\s+(\d+)>", re.IGNORECASE)
_PICTURE_TAG = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)
_VIDEO_TAG = re.compile(r"<Video\s+(\d+)>", re.IGNORECASE)
_SUBJECT_TAG = re.compile(r"<Subject\s+(\d+)>", re.IGNORECASE)
_DIALOGUE_TAG = re.compile(r"<d>", re.IGNORECASE)
_SPEAKER_TAG = re.compile(r"\(S(\d+)\)", re.IGNORECASE)

# A declaration binds a reference label to what it depicts. Everything else in
# the prompt is the segment body: the part that decides what is actually used.
_DECLARATION_LINE = re.compile(
    r"^\s*<(?:Subject|Picture|Video|Audio)\s+\d+>\s*(?:\(S\d+\)\s*)?(?:is\b|:)",
    re.IGNORECASE,
)
_AUDIO_SPEAKER = re.compile(
    r"<Audio\s+(\d+)>[^\n]{0,240}?\(S(\d+)\)", re.IGNORECASE
)
_AUDIO_SUBJECT = re.compile(
    r"<Audio\s+(\d+)>[^\n]{0,240}?<Subject\s+(\d+)>", re.IGNORECASE
)
_SUBJECT_SPEAKER = re.compile(r"<Subject\s+(\d+)>\s*\(S(\d+)\)", re.IGNORECASE)

# How far back a <d> block may look for the speaker tag that owns it.
_SPEAKER_BACKTRACK_WINDOW = 400


def _audio_speaker_map(prompt):
    """Map each source audio label to the speaker ID it voices.

    Accepts both declaration styles these scripts use, `<Audio 1> is the
    voice-timbre reference for <Subject 1> (S1)` and `<Audio 1>: reference -
    timbre guides <Subject 1> (S1)`, and falls back to resolving the speaker
    through the subject when the audio declaration itself names no `(Sn)`.
    """
    mapping = {int(audio): int(speaker) for audio, speaker in _AUDIO_SPEAKER.findall(prompt)}
    subject_speaker = {
        int(subject): int(speaker) for subject, speaker in _SUBJECT_SPEAKER.findall(prompt)
    }
    for audio, subject in _AUDIO_SUBJECT.findall(prompt):
        audio, subject = int(audio), int(subject)
        if audio not in mapping and subject in subject_speaker:
            mapping[audio] = subject_speaker[subject]
    return mapping


def _segment_body(prompt):
    """The prompt minus its reference declaration lines."""
    return "\n".join(
        line for line in prompt.splitlines() if not _DECLARATION_LINE.match(line)
    )


def _spoken_speakers(prompt):
    """Speakers that actually deliver a `<d>` line in this segment.

    Each `<d>` backtracks to the nearest preceding `(Sn)` tag, which is how
    these scripts already attribute dialogue, so no speech-verb vocabulary is
    involved. Declaration lines are removed first: the `(Sn)` in `<Audio 2> is
    the voice of <Subject 2> (S2)` defines a speaker ID, it does not hand that
    speaker a line. An empty set means "cannot attribute" - either there is no
    dialogue or some line has no speaker in reach - and callers treat that as
    "keep every voice".
    """
    body = _segment_body(prompt)
    marks = [(match.end(), int(match.group(1))) for match in _SPEAKER_TAG.finditer(body)]
    speakers = set()
    for dialogue in _DIALOGUE_TAG.finditer(body):
        start = dialogue.start()
        preceding = [mark for mark in marks if mark[0] <= start]
        if not preceding or start - preceding[-1][0] > _SPEAKER_BACKTRACK_WINDOW:
            return set()
        speakers.add(preceding[-1][1])
    return speakers


def _subject_reference_map(prompt):
    """Pictures/videos each declaration ties to one or more `<Subject N>`.

    Both directions are declarations of the same binding, so co-occurrence on a
    declaration line is enough: `<Subject 1> is the man in <Picture 1>` and
    `<Picture 1> is the identity reference for <Subject 1>`.
    """
    pictures = {}
    videos = {}
    for line in prompt.splitlines():
        if not _DECLARATION_LINE.match(line):
            continue
        subjects = {int(label) for label in _SUBJECT_TAG.findall(line)}
        if not subjects:
            continue
        for label in {int(label) for label in _PICTURE_TAG.findall(line)}:
            pictures.setdefault(label, set()).update(subjects)
        for label in {int(label) for label in _VIDEO_TAG.findall(line)}:
            videos.setdefault(label, set()).update(subjects)
    return pictures, videos


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
    """Select voice references from the dialogue the segment actually contains.

    A voice reference only does anything when someone speaks, so a `<d>` block
    is what turns audio references on - not an `<Audio N>` tag. Each `<d>`
    backtracks to its `(Sn)` speaker, and only the voices of speakers that
    reach a line survive. Anything that cannot be attributed is retained, so
    ambiguous prose never silently drops a user-provided reference, and an
    explicit "not used" always wins.
    """
    active = {label for label in available_labels if not _explicitly_inactive(prompt, label)}
    if not active or not _DIALOGUE_TAG.search(prompt):
        # Nothing is spoken here: every voice reference is dead weight.
        return set()

    speakers = _spoken_speakers(prompt)
    speaker_map = _audio_speaker_map(prompt)
    if not speakers or not speaker_map:
        return active
    return {
        label for label in active
        if speaker_map.get(label) is None or speaker_map[label] in speakers
    }


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
    """Select image/video refs named by the segment, pruned by subject usage.

    A `<Picture N>` declared as the reference *for* a `<Subject M>` is only
    worth its packed rows when that subject is used in the segment body, so a
    declaration block listing every character no longer drags every identity
    image into every segment. Pruning is skipped entirely when the body names
    no subject at all, because then the prompt is not written in subject style
    and dropping references would lose identity rather than save context.
    """
    pictures = {
        int(label) for label in _PICTURE_TAG.findall(prompt)
        if int(label) in available_pictures
    }
    videos = {
        int(label) for label in _VIDEO_TAG.findall(prompt)
        if int(label) in available_videos
    }

    used_subjects = {int(label) for label in _SUBJECT_TAG.findall(_segment_body(prompt))}
    if not used_subjects:
        return pictures, videos

    picture_subjects, video_subjects = _subject_reference_map(prompt)

    def keep(label, subject_map):
        subjects = subject_map.get(label)
        return not subjects or bool(subjects & used_subjects)

    pictures = {label for label in pictures if keep(label, picture_subjects)}
    videos = {label for label in videos if keep(label, video_subjects)}
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
