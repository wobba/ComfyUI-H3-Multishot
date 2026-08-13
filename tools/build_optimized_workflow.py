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
            8,
            False,
            "system_ram",
            True,
            False,
            False,
            True,
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
    nodes[16] = spectrum_node(16, "FL2VA Spectrum", 1, 2, -1010, -220)
    nodes[17] = sage_node(17, "FL2VA SageAttention3", 2, 3, -560, -220)

    nodes[18] = model_node(
        18,
        "Ref2VA INT8 - only routed native-ref segments",
        "minimax_h3_ref2va_int8_convrot.safetensors",
        4,
        -1460,
        430,
    )
    nodes[20] = spectrum_node(20, "Ref2VA Spectrum", 4, 5, -1010, 430)
    nodes[21] = sage_node(21, "Ref2VA SageAttention3", 5, 6, -560, 430)

    sampler = nodes[6]
    sampler["pos"] = [300, -120]
    sampler["size"] = [590, 720]
    sampler["title"] = "Optimized Variable-Duration Memory Chain"
    sampler["inputs"] = [
        {"name": "model", "type": "MODEL", "link": 3},
        {"name": "clip", "type": "CLIP", "link": 7},
        {"name": "video_vae", "type": "VAE", "link": 8},
        {"name": "audio_vae", "type": "VAE", "link": 9},
        {"name": "persistent_refs", "type": "H3_REFS", "link": 12},
        {"name": "ref2va_model", "type": "MODEL", "link": 6},
    ]
    sampler["outputs"][0]["links"] = [17]
    sampler["outputs"][1]["links"] = [18]
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
        "res_multistep",
        "simple",
        "auto_speaker_aware",
        "",
        "",
        "auto_prompt_aware",
        "",
    ]

    nodes[2]["pos"] = [-1460, 1080]
    nodes[3]["pos"] = [-1010, 1080]
    nodes[4]["pos"] = [-1010, 1180]
    nodes[5]["pos"] = [-550, 1080]
    nodes[5]["outputs"][0]["links"] = [12]
    nodes[5]["widgets_values"] = [960, 544, 243, "match", 12]
    bank_links = {
        "video_vae": 10,
        "audio_vae": 11,
        "ref_image_1": 13,
        "ref_image_2": 14,
        "ref_image_3": 15,
        "ref_audio_1": 16,
    }
    for input_spec in nodes[5]["inputs"]:
        input_spec["link"] = bank_links[input_spec["name"]]
    nodes[7]["inputs"][0]["link"] = 17
    nodes[7]["inputs"][1]["link"] = 18
    nodes[7]["outputs"][0]["links"] = [19]
    nodes[8]["inputs"][0]["link"] = 19

    nodes[2]["outputs"][0]["links"] = [7]
    nodes[3]["outputs"][0]["links"] = [8, 10]
    nodes[4]["outputs"][0]["links"] = [9, 11]
    nodes[10]["outputs"][0]["links"] = [13]
    nodes[11]["outputs"][0]["links"] = [14]
    nodes[12]["outputs"][0]["links"] = [15]
    nodes[13]["outputs"][0]["links"] = [16]
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
            "Both INT8 models are patched once through **Spectrum -> "
            "SageAttention3**. The sampler uses FL2VA for a segment with no "
            "routed native reference blocks and Ref2VA only when that segment "
            "uses `<Picture N>`, `<Video N>`, or active audio refs.\n\n"
            "For a content or Turbo LoRA, insert Larryvrh's MiniMax-H3 Turbo "
            "LoRA loader before Spectrum on the model path that needs it. The "
            "baseline has no LoRA because Spectrum and SageAttention3 provide "
            "the measured speedup.\n\n"
            "Put `frame_count: N` inside each outer `---` prompt block. The "
            "sampler strips it before conditioning and uses it for that "
            "segment's latent length. A block without it uses "
            "`frames_per_shot`. Internal "
            "`[Shot N] At MM:SS.mmm` camera cuts can occur at arbitrary times "
            "inside a segment and do not need to match generation boundaries.\n\n"
            "Reference loaders are muted by default. Enable and select only the "
            "assets required by the script. `auto_prompt_aware` compacts visual "
            "labels per segment. Spectrum's default offline replay uses a second "
            "sampler pass but was retained because it matches the validated fast "
            "T2V workflow."
        )
    ]

    workflow["id"] = "h3-memory-optimized-int8-variable"
    workflow["last_node_id"] = 21
    workflow["last_link_id"] = 19
    workflow["nodes"] = [nodes[node_id] for node_id in sorted(nodes)]
    workflow["links"] = [
        [1, 14, 0, 16, 0, "MODEL"],
        [2, 16, 0, 17, 0, "MODEL"],
        [3, 17, 0, 6, 0, "MODEL"],
        [4, 18, 0, 20, 0, "MODEL"],
        [5, 20, 0, 21, 0, "MODEL"],
        [6, 21, 0, 6, 18, "MODEL"],
        [7, 2, 0, 6, 1, "CLIP"],
        [8, 3, 0, 6, 2, "VAE"],
        [9, 4, 0, 6, 3, "VAE"],
        [10, 3, 0, 5, 0, "VAE"],
        [11, 4, 0, 5, 1, "VAE"],
        [12, 5, 0, 6, 17, "H3_REFS"],
        [13, 10, 0, 5, 7, "IMAGE"],
        [14, 11, 0, 5, 8, "IMAGE"],
        [15, 12, 0, 5, 9, "IMAGE"],
        [16, 13, 0, 5, 22, "AUDIO"],
        [17, 6, 0, 7, 0, "IMAGE"],
        [18, 6, 1, 7, 1, "AUDIO"],
        [19, 7, 0, 8, 0, "VIDEO"],
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
