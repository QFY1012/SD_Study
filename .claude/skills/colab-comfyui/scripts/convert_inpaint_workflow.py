#!/usr/bin/env python3
"""Turn an inpaint PNG's embedded API workflow into one that runs on a remote-only ComfyUI.

The original workflows load CLIP and VAE from local files (DualCLIPLoader / VAELoader). On the
remote machine only the full checkpoint exists, so both loaders are replaced by one
CheckpointLoaderSimple and every link to them is rewired to its CLIP (slot 1) / VAE (slot 2) outputs.

usage: convert_inpaint_workflow.py SRC.png OUT.json --ckpt xaviernocturne_v3.safetensors \
           [--image NAME_IN_INPUT_DIR] [--prefix OUTPUT_PREFIX]
"""
import argparse
import json
import sys

from PIL import Image


def is_link(v):
    return (isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))
            and isinstance(v[1], int) and not isinstance(v[0], bool))


def convert(prompt, ckpt, image=None, prefix=None):
    p = json.loads(json.dumps(prompt))  # deep copy
    clip_ids = [k for k, n in p.items() if n["class_type"] in ("DualCLIPLoader", "CLIPLoader")]
    vae_ids = [k for k, n in p.items() if n["class_type"] == "VAELoader"]
    new_id = "ckpt_loader"
    p[new_id] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt},
                 "_meta": {"title": "Load Checkpoint (CLIP + VAE from checkpoint)"}}
    for node in p.values():
        for name, v in list(node.get("inputs", {}).items()):
            if isinstance(v, list) and len(v) == 2:
                if v[0] in clip_ids:
                    node["inputs"][name] = [new_id, 1]
                elif v[0] in vae_ids:
                    node["inputs"][name] = [new_id, 2]
    for k in clip_ids + vae_ids:
        del p[k]
    for k, n in p.items():
        n.pop("is_changed", None)
        if image and n["class_type"] == "LoadImage":
            n["inputs"]["image"] = image
        if prefix and n["class_type"] == "SaveImage":
            n["inputs"]["filename_prefix"] = prefix
        if n["class_type"] == "RemoteSEGSDetailer":
            n["inputs"]["model_name"] = ckpt
    # A plain list in an API input means "link to another node". List-valued *widget* values
    # (e.g. CreateBoundingBoxes.editor_state) must be wrapped as {"__value__": [...]} (see
    # ComfyUI execution.py), otherwise validation treats a 2-element list as a broken link.
    for n in p.values():
        for name, v in list(n.get("inputs", {}).items()):
            if isinstance(v, list) and not is_link(v):
                n["inputs"][name] = {"__value__": v}
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_png")
    ap.add_argument("out_json")
    ap.add_argument("--ckpt", default="xaviernocturne_v3.safetensors")
    ap.add_argument("--image")
    ap.add_argument("--prefix")
    a = ap.parse_args()
    info = Image.open(a.src_png).info
    if "prompt" not in info:
        sys.exit("PNG has no embedded 'prompt' metadata")
    out = convert(json.loads(info["prompt"]), a.ckpt, a.image, a.prefix)
    json.dump(out, open(a.out_json, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {a.out_json}: {len(out)} nodes")


if __name__ == "__main__":
    main()
