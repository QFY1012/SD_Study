import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

DEFAULT_SD_DIR = Path(r"D:\untig\SD_Study")
DEFAULT_COMFY_ROOT = Path(r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI")
DEFAULT_SAM3_CKPT = DEFAULT_COMFY_ROOT / "models" / "checkpoints" / "sam3.1_multiplex_fp16.safetensors"
DEFAULT_DINOV3_DIR = DEFAULT_SD_DIR / "models" / "dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_HEAD_PT = DEFAULT_SD_DIR / "models" / "dinov3_hand_spatial2d_head.pt"


class Spatial2DRewardHead(nn.Module):
    """2D spatial conv reward head over DINOv3 CLS + 24x24 patch grid."""

    def __init__(self, dim: int = 768, ch: int = 64):
        super().__init__()
        self.cls_head = nn.Linear(dim, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, ch, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(ch * 2, 1),
        )

    def forward(self, cls_vec: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        h = self.conv(grid.float())
        avg_p = h.mean(dim=(2, 3))
        max_p = h.amax(dim=(2, 3))
        return self.cls_head(cls_vec.float()) + self.out(torch.cat([avg_p, max_p], dim=-1))


def make_hand_crop(img: Image.Image, boxes: list[dict], pad_frac: float = 0.15):
    """Crop top-1 or top-2 detected hand boxes into a single crop image."""
    W, H = img.size
    if not boxes:
        return None
    top_boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)[:2]
    crops = []
    for b in top_boxes:
        x1, y1, x2, y2 = b["box"]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx1 = max(0, int(x1 - bw * pad_frac))
        cy1 = max(0, int(y1 - bh * pad_frac))
        cx2 = min(W, int(x2 + bw * pad_frac))
        cy2 = min(H, int(y2 + bh * pad_frac))
        if cx2 > cx1 and cy2 > cy1:
            crops.append(img.crop((cx1, cy1, cx2, cy2)))
    if not crops:
        return None
    if len(crops) == 1:
        return crops[0]
    c1 = crops[0].resize((192, 384), Image.Resampling.LANCZOS)
    c2 = crops[1].resize((192, 384), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (384, 384))
    canvas.paste(c1, (0, 0))
    canvas.paste(c2, (192, 0))
    return canvas


def run_sam3_crops(
    img_paths: list[Path],
    crop_dir: Path,
    bbox_json: Path,
    comfy_root: Path = DEFAULT_COMFY_ROOT,
    ckpt_path: Path = DEFAULT_SAM3_CKPT,
    text_prompt: str = "hand",
    threshold: float = 0.30,
    prefix_parent: bool = False,
) -> dict:
    """Incrementally detect hand bboxes with ComfyUI SAM3.1 and save crops."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    bbox_json.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(bbox_json.read_text(encoding="utf-8")) if bbox_json.is_file() else {}
    todo = [str(p.resolve()) for p in img_paths if str(p.resolve()) not in meta]
    print(f"SAM3 total={len(img_paths)}, cached={len(meta)}, todo={len(todo)}", flush=True)
    if not todo:
        return meta

    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    import folder_paths
    import comfy.sd
    import comfy.model_management
    import comfy.utils

    t0 = time.time()
    out = comfy.sd.load_checkpoint_guess_config(
        str(ckpt_path),
        output_vae=False,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, clip = out[0], out[1]
    tokens = clip.tokenize(text_prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model

    emb = cond[0][0].to(device=device, dtype=dtype).detach()
    mask = cond[0][1].get("attention_mask")
    mask = mask.to(device).detach() if mask is not None else torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
    del clip, out, tokens, cond
    gc.collect()
    torch.cuda.empty_cache()
    print(f"SAM3 ready in {time.time()-t0:.1f}s (VRAM={torch.cuda.memory_allocated()/1e6:.0f}MB)", flush=True)

    t_loop = time.time()
    for idx, p in enumerate(todo, 1):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        image_in = comfy.utils.common_upscale(arr.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        with torch.inference_mode():
            res = sam3_model(
                image_in.to(device=device, dtype=dtype),
                text_embeddings=emb,
                text_mask=mask,
                boxes=None,
                threshold=threshold,
                orig_size=(H, W),
            )
            probs = res["scores"][0].sigmoid()
            keep = probs > threshold
            boxes = res["boxes"][0][keep].cpu().tolist()
            scores = probs[keep].cpu().tolist()
        del res, probs, keep, image_in, arr
        torch.cuda.empty_cache()

        det = [{"box": b, "score": float(s)} for b, s in zip(boxes, scores)]
        crop_img = make_hand_crop(img, det)
        crop_rel = None
        if crop_img is not None:
            name = (Path(p).parent.name + "__" + Path(p).stem + ".png") if prefix_parent else (Path(p).stem + ".png")
            cp = crop_dir / name
            crop_img.save(cp)
            crop_rel = str(cp)
        meta[p] = {"boxes": det, "crop": crop_rel}
        if idx % 20 == 0 or idx == len(todo):
            bbox_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t_loop
            eta = el / idx * (len(todo) - idx)
            print(f"SAM3 [{idx}/{len(todo)}] elapsed={el:.1f}s | ETA={eta:.1f}s", flush=True)

    del sam3_model, model, emb, mask
    gc.collect()
    torch.cuda.empty_cache()
    bbox_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def extract_spatial24(
    image_paths: list[str],
    device: torch.device,
    model_dir: Path = DEFAULT_DINOV3_DIR,
    batch_size: int = 4,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Extract normalized CLS [768] + 24x24 patch grid [768, 24, 24] in fp16."""
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()

    feats = {}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=imgs, size={"height": 384, "width": 384}, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
            out = model(**inputs)
            cls_tok = F.normalize(out.pooler_output.float(), dim=-1).half().cpu()
            patches = out.last_hidden_state[:, 5:, :]
            patches = F.normalize(patches.float(), dim=-1).view(-1, 24, 24, 768).permute(0, 3, 1, 2).contiguous().half().cpu()
            for p, c, g in zip(batch_paths, cls_tok, patches):
                feats[p] = (c, g)
    print(f"Extracted spatial 24x24 for {len(image_paths)} crops in {time.time()-t0:.1f}s", flush=True)
    return feats
