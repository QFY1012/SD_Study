"""ComfyUI-SEGSInpaint

Standalone (no Impact Pack, no remote plugin) replacements for the two nodes used by the
supervised-inpaint workflows:

  * SAM3BoxesToSEGS     - drawn boxes -> SEGS (the box itself is the mask, no SAM needed)
  * RemoteSEGSDetailer  - per-SEGS crop -> upscale to guide_size -> masked img2img -> paste back

Class names and input names are kept identical to the original workflows so the API-format
workflow embedded in the generated PNGs can be loaded and run unchanged. `url` and `token`
are accepted for compatibility and ignored: everything runs on this ComfyUI instance.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from geometry import boxes_from_input, expand_box, crop_region_for, target_size  # noqa: E402

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes as comfy_nodes


# ----------------------------------------------------------------------------- helpers
def gaussian_blur_mask(mask, radius):
    """mask: [1,1,H,W] float. Separable gaussian, radius in pixels (0 = no-op)."""
    radius = int(radius)
    if radius <= 0:
        return mask
    sigma = max(radius / 2.0, 0.5)
    k = torch.arange(-radius, radius + 1, dtype=mask.dtype, device=mask.device)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    pad = radius
    x = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x.clamp(0, 1)


_MODEL_CACHE = {}


def load_base_model(model_name):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # keep a single checkpoint resident
        path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
        out = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=False, output_clip=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        _MODEL_CACHE[model_name] = out[0]
    return _MODEL_CACHE[model_name]


def apply_loras(model, loras_json):
    try:
        entries = json.loads(loras_json) if loras_json and loras_json.strip() else []
    except json.JSONDecodeError as e:
        raise ValueError(f"`loras` is not valid JSON: {e}")
    for entry in entries:
        strength = float(entry.get("strength", 0))
        if not entry.get("name") or strength == 0:
            continue
        path = folder_paths.get_full_path_or_raise("loras", entry["name"])
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
    return model


# ----------------------------------------------------------------------------- nodes
class SAM3BoxesToSEGS:
    """Boxes -> SEGS. `padding` grows each drawn box (the mask); `crop_factor` scales that
    padded box about its centre to get the context crop that is actually inpainted."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bboxes": ("BOUNDING_BOX",),
            "crop_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            "padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "convert"
    CATEGORY = "image/detailer"

    def convert(self, bboxes, crop_factor, padding, image):
        _, h, w, _ = image.shape
        segs = []
        for box in boxes_from_input(bboxes, w, h):
            mask_box = expand_box(box, padding, w, h)
            region = crop_region_for(mask_box, crop_factor, w, h)
            rx1, ry1, rx2, ry2 = region
            mask = torch.zeros((ry2 - ry1, rx2 - rx1), dtype=torch.float32)
            mask[mask_box[1] - ry1:mask_box[3] - ry1, mask_box[0] - rx1:mask_box[2] - rx1] = 1.0
            segs.append({"bbox": mask_box, "crop_region": region, "mask": mask})
        if not segs:
            print("[SEGSInpaint] SAM3BoxesToSEGS: no valid boxes, detailer will be a no-op")
        return ({"shape": (h, w), "segs": segs},)


class RemoteSEGSDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
            "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
            "feather": ("INT", {"default": 5, "min": 0, "max": 100}),
            "noise_mask": ("BOOLEAN", {"default": True}),
            "force_inpaint": ("BOOLEAN", {"default": True}),
            "cycle": ("INT", {"default": 1, "min": 1, "max": 10}),
            "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100}),
            "tiled_encode": ("BOOLEAN", {"default": False}),
            "tiled_decode": ("BOOLEAN", {"default": False}),
            "model_name": (folder_paths.get_filename_list("checkpoints"),),
            "loras": ("STRING", {"default": "[]", "multiline": True}),
            "image": ("IMAGE",),
            "segs": ("SEGS",),
            "vae": ("VAE",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }, "optional": {
            # accepted for workflow compatibility, ignored (everything runs locally)
            "url": ("STRING", {"default": ""}),
            "token": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detail"
    CATEGORY = "image/detailer"

    # -- core: one crop through VAE-encode / sample / VAE-decode -------------------------
    def _inpaint_crop(self, crop, mask, new_w, new_h, model, vae, positive, negative, seed, steps,
                      cfg, sampler_name, scheduler, denoise, use_noise_mask, noise_mask_feather,
                      tiled_encode, tiled_decode, cycle, upscale):
        pix = F.interpolate(crop.permute(0, 3, 1, 2), size=(new_h, new_w), mode="bicubic",
                            align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)[..., :3]
        m = F.interpolate(mask[None, None], size=(new_h, new_w), mode="bilinear", align_corners=False)
        m = gaussian_blur_mask(m, round(noise_mask_feather * upscale))
        for _ in range(cycle):
            latent = vae.encode_tiled(pix) if tiled_encode else vae.encode(pix)
            lat = {"samples": latent}
            if use_noise_mask:
                lat["noise_mask"] = m
            out = comfy_nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                              positive, negative, lat, denoise=denoise)[0]
            samples = out["samples"]
            pix = vae.decode_tiled(samples) if tiled_decode else vae.decode(samples)
            if pix.dim() == 5:
                pix = pix.reshape(-1, *pix.shape[-3:])
            pix = pix[..., :3].clamp(0, 1)
        return pix

    def detail(self, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
               denoise, feather, noise_mask, force_inpaint, cycle, noise_mask_feather, tiled_encode,
               tiled_decode, model_name, loras, image, segs, vae, positive, negative, url="", token=""):
        seg_list = segs["segs"]
        if not seg_list:
            return (image,)
        model = apply_loras(load_base_model(model_name), loras)
        result = image.clone()
        for b in range(result.shape[0]):
            for i, seg in enumerate(seg_list):
                rx1, ry1, rx2, ry2 = seg["crop_region"]
                bx1, by1, bx2, by2 = seg["bbox"]
                crop = result[b:b + 1, ry1:ry2, rx1:rx2, :]
                ch, cw = crop.shape[1], crop.shape[2]
                ref_w, ref_h = (bx2 - bx1, by2 - by1) if guide_size_for else (cw, ch)
                new_w, new_h, upscale = target_size(ref_w, ref_h, cw, ch, guide_size, max_size)
                if upscale <= 1.0 and not force_inpaint:
                    print(f"[SEGSInpaint] seg {i}: already >= guide_size, skipped")
                    continue
                mask = seg["mask"].to(crop.device)
                new_pix = self._inpaint_crop(crop, mask, new_w, new_h, model, vae, positive, negative,
                                             seed, steps, cfg, sampler_name, scheduler, denoise,
                                             noise_mask, noise_mask_feather, tiled_encode, tiled_decode,
                                             cycle, upscale)
                new_pix = F.interpolate(new_pix.permute(0, 3, 1, 2), size=(ch, cw), mode="bicubic",
                                        align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)
                blend = gaussian_blur_mask(mask[None, None], feather)[0, 0][None, :, :, None]
                result[b:b + 1, ry1:ry2, rx1:rx2, :] = crop * (1 - blend) + new_pix.to(crop.dtype) * blend
        return (result,)


NODE_CLASS_MAPPINGS = {"SAM3BoxesToSEGS": SAM3BoxesToSEGS, "RemoteSEGSDetailer": RemoteSEGSDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3BoxesToSEGS": "Boxes → SEGS (no SAM)",
    "RemoteSEGSDetailer": "SEGS Detailer (local, drop-in)",
}
