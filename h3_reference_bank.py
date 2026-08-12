# -*- coding: utf-8 -*-
"""Persistent native MiniMax-H3 references for chained multishot workflows."""

import math
import os


MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3


def _model_choices(variant=None):
    import folder_paths

    files = set(folder_paths.get_filename_list("diffusion_models"))
    for directory in folder_paths.get_folder_paths("diffusion_models"):
        if not os.path.isdir(directory):
            continue
        for root, _directories, names in os.walk(directory):
            for name in names:
                if name.lower().endswith(".gguf"):
                    files.add(os.path.relpath(os.path.join(root, name), directory))
    files = sorted(files)
    if variant:
        # Community GGUF packs commonly use hyphenated names such as
        # MiniMax-H3-FL2VA-Q5_K_M.gguf, while the official safetensors names
        # use underscores. Both are the same H3 variant and H3ModelLoaderAny
        # can load either one.
        needle = f"minimax_h3_{variant}"
        matching = [
            name for name in files
            if needle in name.lower().replace("-", "_")
        ]
        if matching:
            return matching
    return files


def _prepare_audio(audio_vae, audio, max_seconds):
    import torchaudio

    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    waveform = waveform[:1]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]

    sample_rate = int(audio["sample_rate"])
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
        sample_rate = vae_sample_rate

    max_samples = int(max_seconds * sample_rate)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    latent = audio_vae.encode(waveform.movedim(1, -1))
    return {
        "kind": "audio",
        "ref_audio_t": latent.shape[-1],
        "audio_latent": latent,
    }, waveform.shape[-1] / sample_rate


class H3PersistentReferenceBank:
    """Prepare native Ref2VA references once and reuse them for every shot."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "max_audio_seconds": (
                "FLOAT",
                {
                    "default": 12.0,
                    "min": 0.5,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Each reference audio row participates in every denoising step. "
                    "Use a clean 6-12 second voice clip; 15 seconds is the H3 limit.",
                },
            ),
        }
        for index in range(1, MAX_IMAGES + 1):
            optional[f"ref_image_{index}"] = (
                "IMAGE",
                {"tooltip": f"Persistent native image reference <Picture {index}>."},
            )
        for index in range(1, MAX_VIDEOS + 1):
            optional[f"ref_video_{index}"] = (
                "IMAGE",
                {
                    "tooltip": f"Reference video {index} as an IMAGE frame batch. "
                    "Use a video loader that outputs frames at 24 fps.",
                },
            )
            optional[f"ref_video_audio_{index}"] = (
                "AUDIO",
                {"tooltip": f"Optional soundtrack paired with reference video {index}."},
            )
        for index in range(1, MAX_AUDIOS + 1):
            optional[f"ref_audio_{index}"] = (
                "AUDIO",
                {"tooltip": f"Persistent standalone audio reference <Audio {index}>."},
            )

        return {
            "required": {
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 4096, "step": 32}),
                "frames_per_shot": (
                    "INT",
                    {
                        "default": 243,
                        "min": 5,
                        "max": 1000,
                        "step": 17,
                        "tooltip": "Used to clip and align reference videos to the target H3 duration.",
                    },
                ),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": "Use match on 32 GB cards. Max preserves more image detail "
                        "but increases every-shot attention cost.",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("H3_REFS", "BOOLEAN", "STRING")
    RETURN_NAMES = ("persistent_refs", "has_references", "reference_report")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax"

    def build(
        self,
        video_vae,
        audio_vae,
        width,
        height,
        frames_per_shot,
        ref_image_size,
        max_audio_seconds=12.0,
        **references,
    ):
        from comfy_extras import nodes_minimax_h3 as mmh3

        frame_count = mmh3.align_frame_count(max(5, frames_per_shot))
        items = []
        blocks = []
        entries = []
        report = []

        for index in range(1, MAX_IMAGES + 1):
            image = references.get(f"ref_image_{index}")
            if image is None:
                continue
            image = image[:1]
            height_in, width_in = image.shape[1:3]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (width_in * height_in)))
            else:
                scale = min(1.0, mmh3.REF_IMAGE_SHORT_EDGE / min(width_in, height_in))
            target_width = max(
                mmh3.CANVAS_MULTIPLE,
                round(width_in * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE,
            )
            target_height = max(
                mmh3.CANVAS_MULTIPLE,
                round(height_in * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE,
            )
            resized = mmh3._resize(image, target_width, target_height, "disabled")
            latent = video_vae.encode(resized)
            item = {"type": "image", "data": resized}
            block = {
                "kind": "image",
                "latent_h": target_height // 16,
                "latent_w": target_width // 16,
                "latent": latent,
            }
            items.append(item)
            blocks.append(block)
            entries.append({"kind": "image", "items": [item], "blocks": [block]})
            report.append(f"Picture {sum(item['type'] == 'image' for item in items)}")

        for index in range(1, MAX_VIDEOS + 1):
            video = references.get(f"ref_video_{index}")
            if video is None:
                continue
            video_height, video_width = video.shape[1:3]
            target_width, target_height = mmh3.adapt_canvas(video_width, video_height)
            if video_width * video_height < target_width * target_height:
                target_width = max(
                    mmh3.CANVAS_MULTIPLE,
                    round(video_width / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE,
                )
                target_height = max(
                    mmh3.CANVAS_MULTIPLE,
                    round(video_height / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE,
                )
            frames = mmh3._resize(video, target_width, target_height, "disabled")[:frame_count]
            if frames.shape[0] < 5:
                raise ValueError(
                    f"Reference video {index} needs at least five frames before H3 frame-grid alignment."
                )
            while frames.shape[0] % 17 != 5:
                frames = frames[:-1]

            audio_block = None
            audio_item = None
            audio_label = None
            paired_audio = references.get(f"ref_video_audio_{index}")
            if paired_audio is not None:
                audio_block, seconds = _prepare_audio(audio_vae, paired_audio, max_audio_seconds)
                audio_item = {"type": "audio"}
                items.append(audio_item)
                audio_label = sum(item["type"] == "audio" for item in items)
                report.append(f"Audio {audio_label} (video {index})")

            latent = video_vae.encode(frames)
            sample_indices = list(range(0, frames.shape[0], mmh3.FPS // 2))
            video_item = {
                "type": "video",
                "data": frames[sample_indices],
                "timestamps": [i / 2.0 for i in range(len(sample_indices))],
            }
            block = {
                "kind": "video_audio" if audio_block else "video",
                "latent_t": latent.shape[2],
                "latent_h": target_height // 16,
                "latent_w": target_width // 16,
                "ref_audio_t": audio_block["ref_audio_t"] if audio_block else 0,
                "latent": latent,
                "audio_latent": audio_block["audio_latent"] if audio_block else None,
            }
            items.append(video_item)
            blocks.append(block)
            entries.append(
                {
                    "kind": "video",
                    "items": ([audio_item] if audio_item else []) + [video_item],
                    "blocks": [block],
                    "video_item": video_item,
                    "audio_label": audio_label,
                }
            )
            report.append(f"Video {index}")

        for index in range(1, MAX_AUDIOS + 1):
            audio = references.get(f"ref_audio_{index}")
            if audio is None:
                continue
            block, seconds = _prepare_audio(audio_vae, audio, max_audio_seconds)
            item = {"type": "audio"}
            items.append(item)
            blocks.append(block)
            audio_label = sum(item["type"] == "audio" for item in items)
            entries.append(
                {
                    "kind": "audio",
                    "items": [item],
                    "blocks": [block],
                    "audio_label": audio_label,
                }
            )
            report.append(f"Audio {audio_label} ({seconds:.1f}s)")

        bank = {
            "items": items,
            "blocks": blocks,
            "entries": entries,
            "has_references": bool(blocks),
            "report": ", ".join(report) if report else "No persistent references",
        }
        print(f"[H3PersistentRefs] {bank['report']}", flush=True)
        return (bank, bank["has_references"], bank["report"])


class H3ReferenceAwareModelLoader:
    """Select FL2VA or Ref2VA based on whether a persistent reference bank is populated."""

    @classmethod
    def INPUT_TYPES(cls):
        fl2va_choices = _model_choices("fl2va")
        ref2va_choices = _model_choices("ref2va")
        return {
            "required": {
                "fl2va_model": (
                    fl2va_choices,
                    {
                        "default": "minimax_h3_fl2va_pruned_nvfp4.safetensors",
                        "tooltip": "H3 FL2VA safetensors or GGUF model used when "
                                   "the connected reference bank is empty.",
                    },
                ),
                "ref2va_model": (
                    ref2va_choices,
                    {
                        "default": "minimax_h3_ref2va_pruned_nvfp4.safetensors",
                        "tooltip": "H3 Ref2VA safetensors or GGUF model used whenever "
                                   "the connected reference bank contains images, video, or audio.",
                    },
                ),
            },
            "optional": {
                "persistent_refs": ("H3_REFS",),
                "activation_reserve_gb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.5,
                        "tooltip": "0 uses the pack's adaptive per-shape reserve.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN")
    RETURN_NAMES = ("model", "selected_model", "using_ref2va")
    FUNCTION = "load"
    CATEGORY = "loaders/minimax"

    def load(self, fl2va_model, ref2va_model, persistent_refs=None, activation_reserve_gb=0.0):
        from .h3_multishot_utils import H3ModelLoaderAny

        use_ref2va = bool(persistent_refs and persistent_refs.get("has_references"))
        model_name = ref2va_model if use_ref2va else fl2va_model
        model = H3ModelLoaderAny().load(model_name, activation_reserve_gb)[0]
        mode = "Ref2VA" if use_ref2va else "FL2VA"
        print(f"[H3ReferenceModel] {mode}: {model_name}", flush=True)
        return (model, model_name, use_ref2va)


NODE_CLASS_MAPPINGS = {
    "H3PersistentReferenceBank": H3PersistentReferenceBank,
    "H3ReferenceAwareModelLoader": H3ReferenceAwareModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PersistentReferenceBank": "H3 Persistent Reference Bank (Ref2VA)",
    "H3ReferenceAwareModelLoader": "H3 Model Loader (auto FL2VA / Ref2VA)",
}
