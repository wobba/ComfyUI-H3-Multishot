"""Build the optimized variable-duration Memory workflow from the refs sample."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workflows" / "H3_Multishot_MEMORY_Persistent_Refs.json"
OUTPUT = ROOT / "workflows" / "H3_Multishot_MEMORY_Optimized_INT8.json"
SAMPLE = ROOT / "samples" / "variable_duration_two_people.txt"


def model_node(node_id, title, model_name, output_link, x, y):
    return {
        "id": node_id,
        "type": "H3ModelLoaderAny",
        "pos": [x, y],
        "size": [410, 100],
        "title": title,
        "inputs": [],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": [output_link]}
        ],
        "widgets_values": [model_name, 0],
    }


def turbo_node(node_id, title, input_link, output_link, x, y):
    return {
        "id": node_id,
        "type": "MiniMaxH3TurboLoRA",
        "pos": [x, y],
        "size": [370, 130],
        "title": title,
        "inputs": [{"name": "model", "type": "MODEL", "link": input_link}],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [output_link]}
        ],
        "widgets_values": [
            "h3-minimax/penis vagina insert.safetensors",
            1,
            False,
        ],
    }


def spectrum_node(node_id, title, input_link, output_link, x, y):
    return {
        "id": node_id,
        "type": "SpectrumApplyMiniMaxH3",
        "pos": [x, y],
        "size": [390, 460],
        "title": title,
        "inputs": [{"name": "model", "type": "MODEL", "link": input_link}],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [output_link]}
        ],
        "widgets_values": [
            True,
            0.5,
            1,
            0.1,
            2,
            0.75,
            1,
            1,
            2,
            False,
            "system_ram",
            True,
            False,
            False,
            False,
            0,
            "system_ram",
            "off",
            0.65,
        ],
    }


def sage_node(node_id, title, input_link, output_link, x, y):
    return {
        "id": node_id,
        "type": "PathchSageAttentionKJ",
        "pos": [x, y],
        "size": [350, 120],
        "title": title,
        "inputs": [{"name": "model", "type": "MODEL", "link": input_link}],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": [output_link]}
        ],
        "widgets_values": ["sageattn3", False],
    }


def model_route_node(x, y):
    return {
        "id": 22,
        "type": "H3ModelRoute",
        "pos": [x, y],
        "size": [420, 150],
        "title": "Load only the H3 model path this run needs",
        "inputs": [
            {"name": "fl2va_model", "type": "MODEL", "link": 4},
            {"name": "ref2va_model", "type": "MODEL", "link": 8},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [20]},
            {"name": "ref2va_model", "type": "MODEL", "links": [21]},
        ],
        "widgets_values": ["FL2VA only"],
    }


def main():
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"] if node["id"] != 1}

    nodes[14] = model_node(
        14,
        "FL2VA INT8 - segments without native refs",
        "minimax_h3_fl2va_int8_convrot.safetensors",
        1,
        -1460,
        -220,
    )
    nodes[15] = turbo_node(15, "FL2VA LoRA", 1, 2, -1010, -220)
    nodes[16] = spectrum_node(16, "FL2VA Spectrum", 2, 3, -600, -220)
    nodes[17] = sage_node(17, "FL2VA SageAttention3", 3, 4, -170, -220)

    nodes[18] = model_node(
        18,
        "Ref2VA INT8 - only routed native-ref segments",
        "minimax_h3_ref2va_int8_convrot.safetensors",
        5,
        -1460,
        430,
    )
    nodes[19] = turbo_node(19, "Ref2VA LoRA", 5, 6, -1010, 430)
    nodes[20] = spectrum_node(20, "Ref2VA Spectrum", 6, 7, -600, 430)
    nodes[21] = sage_node(21, "Ref2VA SageAttention3", 7, 8, -170, 430)
    nodes[22] = model_route_node(300, 700)

    sampler = nodes[6]
    sampler["pos"] = [300, -120]
    sampler["size"] = [590, 720]
    sampler["type"] = "H3MultishotMemoryDiskSampler"
    sampler["title"] = "Optimized Variable-Duration Memory Chain (Disk / Resume)"
    sampler["inputs"] = [
        {"name": "model", "type": "MODEL", "link": 20},
        {"name": "clip", "type": "CLIP", "link": 9},
        {"name": "video_vae", "type": "VAE", "link": 10},
        {"name": "audio_vae", "type": "VAE", "link": 11},
        {"name": "persistent_refs", "type": "H3_REFS", "link": 14},
        {"name": "ref2va_model", "type": "MODEL", "link": 21},
    ]
    sampler["outputs"] = [
        {"name": "video", "type": "VIDEO", "links": [19]},
        {"name": "manifest_path", "type": "STRING", "links": None},
        {"name": "segments_rendered", "type": "INT", "links": None},
    ]
    sampler["widgets_values"] = [
        SAMPLE.read_text(encoding="utf-8"),
        0,
        960,
        544,
        243,
        0,
        "randomize",
        20,
        True,
        2,
        1,
        "",
        "res_multistep",
        "simple",
        "auto_speaker_aware",
        "",
        "",
        "auto_prompt_aware",
        "",
        True,
        False,
        "",
        True,
    ]

    nodes[2]["pos"] = [-1460, 1080]
    nodes[3]["pos"] = [-1010, 1080]
    nodes[4]["pos"] = [-1010, 1180]
    nodes[5]["pos"] = [-550, 1080]
    nodes[5]["outputs"][0]["links"] = [14]
    nodes[5]["widgets_values"] = [960, 544, 243, "match", 12]
    bank_links = {
        "video_vae": 12,
        "audio_vae": 13,
        "ref_image_1": 15,
        "ref_image_2": 16,
        "ref_image_3": 17,
        "ref_audio_1": 18,
    }
    for input_spec in nodes[5]["inputs"]:
        input_spec["link"] = bank_links[input_spec["name"]]
    nodes.pop(7)
    nodes[8]["inputs"][0]["link"] = 19

    nodes[2]["outputs"][0]["links"] = [9]
    nodes[3]["outputs"][0]["links"] = [10, 12]
    nodes[4]["outputs"][0]["links"] = [11, 13]
    nodes[10]["outputs"][0]["links"] = [15]
    nodes[11]["outputs"][0]["links"] = [16]
    nodes[12]["outputs"][0]["links"] = [17]
    nodes[13]["outputs"][0]["links"] = [18]
    for node_id in (10, 11, 12, 13):
        nodes[node_id]["mode"] = 4

    note = nodes[9]
    note["type"] = "Note"
    note["pos"] = [300, -560]
    note["size"] = [760, 390]
    note["title"] = "Optimized variable-duration routing"
    note["widgets_values"] = [
        (
            "## Optimized H3 Memory chain\n\n"
            "Both INT8 models are patched once through **MiniMax-H3 Turbo "
            "LoRA -> Spectrum -> "
            "SageAttention3**. The sampler uses FL2VA for a segment with no "
            "routed native reference blocks and Ref2VA only when that segment "
            "uses `<Picture N>`, `<Video N>`, or active audio refs.\n\n"
            "Set **H3 Model Route** to `FL2VA only`, `Ref2VA only`, or "
            "`Mixed per segment`. It is lazy: unused loader/LoRA/Spectrum/"
            "Sage chains are not evaluated. No rewiring or muting is needed.\n\n"
            "The LoRA nodes reproduce the validated optimized source graph. "
            "Choose the desired LoRA and strength there before rendering.\n\n"
            "Put `frame_count: N` inside each outer `---` prompt block. The "
            "sampler strips it before conditioning and uses it for that "
            "segment's latent length. A block without it uses "
            "`frames_per_shot`. Internal "
            "`[Shot N] At MM:SS.mmm` camera cuts can occur at arbitrary times "
            "inside a segment and do not need to match generation boundaries.\n\n"
            "Reference loaders are muted by default. Enable and select only the "
            "assets required by the script. `auto_prompt_aware` compacts visual "
            "labels per segment. The Disk/Resume sampler writes lossless segment "
            "intermediates and PNG memory checkpoints immediately, then returns "
            "a lazy VIDEO after final assembly. Leave `run_name` empty for a "
            "unique logged name on every queue. If interrupted, paste that name "
            "back into the widget to resume. LoRA identity is fingerprinted "
            "automatically; `plan_tag` is only an optional manual note."
        )
    ]

    workflow["id"] = "h3-memory-optimized-int8-variable"
    workflow["last_node_id"] = 22
    workflow["last_link_id"] = 21
    workflow["nodes"] = [nodes[node_id] for node_id in sorted(nodes)]
    workflow["links"] = [
        [1, 14, 0, 15, 0, "MODEL"],
        [2, 15, 0, 16, 0, "MODEL"],
        [3, 16, 0, 17, 0, "MODEL"],
        [4, 17, 0, 22, 0, "MODEL"],
        [5, 18, 0, 19, 0, "MODEL"],
        [6, 19, 0, 20, 0, "MODEL"],
        [7, 20, 0, 21, 0, "MODEL"],
        [8, 21, 0, 22, 1, "MODEL"],
        [9, 2, 0, 6, 1, "CLIP"],
        [10, 3, 0, 6, 2, "VAE"],
        [11, 4, 0, 6, 3, "VAE"],
        [12, 3, 0, 5, 0, "VAE"],
        [13, 4, 0, 5, 1, "VAE"],
        [14, 5, 0, 6, 4, "H3_REFS"],
        [15, 10, 0, 5, 2, "IMAGE"],
        [16, 11, 0, 5, 3, "IMAGE"],
        [17, 12, 0, 5, 4, "IMAGE"],
        [18, 13, 0, 5, 5, "AUDIO"],
        [19, 6, 0, 8, 0, "VIDEO"],
        [20, 22, 0, 6, 0, "MODEL"],
        [21, 22, 1, 6, 5, "MODEL"],
    ]
    workflow["groups"] = [
        {
            "title": "Optimized FL2VA and Ref2VA model chains",
            "bounding": [-1490, -260, 1710, 1080],
            "color": "#3f5159",
        },
        {
            "title": "Optional persistent references",
            "bounding": [-1490, 1030, 1390, 830],
            "color": "#594f3f",
        },
    ]
    OUTPUT.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
