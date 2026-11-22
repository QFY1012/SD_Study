import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import DEFAULT_DINOV3_DIR, DEFAULT_SD_DIR, Spatial2DRewardHead


class LoRALinear(nn.Module):
    """Lightweight LoRA wrapper around a frozen nn.Linear."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dx = F.linear(F.linear(self.drop(x.float()), self.lora_A), self.lora_B) * self.scaling
        return out + dx.to(out.dtype)


class HandRMv2(nn.Module):
    """DINOv3 (last block LoRA-tuned) + narrow Spatial2D head."""

    def __init__(self, dinov3_dir: Path, lora_r: int = 4, head_ch: int = 16, lora_dropout: float = 0.25):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(str(dinov3_dir))
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Attach lightweight LoRA to attention q/v in the last transformer layer only
        mods = dict(self.backbone.named_modules())
        attached = 0
        attn = next(m for n, m in mods.items() if n.endswith("layer.11.attention"))
        for proj in ("q_proj", "v_proj"):
            setattr(attn, proj, LoRALinear(getattr(attn, proj), r=lora_r, alpha=lora_r * 2.0, dropout=lora_dropout))
            attached += 1
        assert attached == 2, f"Expected 2 LoRA layers, got {attached}"

        self.head = Spatial2DRewardHead(dim=768, ch=head_ch)
        self.n_reg = getattr(self.backbone.config, "num_register_tokens", 4)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def extract_prefix(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            pv = pixel_values.to(self.backbone.embeddings.patch_embeddings.weight.dtype)
            hidden = self.backbone.embeddings(pv)
            pos_emb = self.backbone.rope_embeddings(pv)
            for blk in self.backbone.model.layer[:11]:
                hidden = blk(hidden, position_embeddings=pos_emb)
        return hidden, pos_emb

    def score_from_prefix(self, hidden: torch.Tensor, pos_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone.model.layer[11](hidden, position_embeddings=pos_emb)
        tokens = self.backbone.norm(hidden)
        cls_vec = F.normalize(tokens[:, 0, :].float(), dim=-1)
        patches = tokens[:, 1 + self.n_reg :, :].float()
        side = int(math.isqrt(patches.shape[1]))
        grid = F.normalize(patches, dim=-1).transpose(1, 2).reshape(patches.shape[0], 768, side, side)
        return self.head(cls_vec, grid)

    def score_crops(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden, pos_emb = self.extract_prefix(pixel_values)
        return self.score_from_prefix(hidden, pos_emb)


def build_split_hand_crops(bbox_meta: dict, crop_dir: Path, pad_frac: float = 0.15, size: int = 384) -> dict[str, list[str]]:
    """Crop each detected hand separately (no squishing) and pad to square."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    img_to_crops: dict[str, list[str]] = {}
    created = 0
    for orig_p, rec in bbox_meta.items():
        boxes = sorted(rec.get("boxes", []), key=lambda x: x["score"], reverse=True)
        if not boxes:
            continue
        # Keep top-2 boxes (skip weak second box if score < 0.35)
        kept = [boxes[0]]
        if len(boxes) > 1 and boxes[1]["score"] >= 0.35:
            kept.append(boxes[1])

        p = Path(orig_p)
        stem = f"{p.parent.name}__{p.stem}"
        paths = []
        img = None
        for hi, b in enumerate(kept):
            cp = crop_dir / f"{stem}__h{hi}.png"
            if not cp.is_file():
                if img is None:
                    if not p.is_file():
                        break
                    img = Image.open(p).convert("RGB")
                W, H = img.size
                x1, y1, x2, y2 = b["box"]
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                cx1, cy1 = max(0, int(x1 - bw * pad_frac)), max(0, int(y1 - bh * pad_frac))
                cx2, cy2 = min(W, int(x2 + bw * pad_frac)), min(H, int(y2 + bh * pad_frac))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                hand = ImageOps.pad(img.crop((cx1, cy1, cx2, cy2)), (size, size), method=Image.Resampling.LANCZOS, color=(128, 128, 128))
                hand.save(cp)
                created += 1
            paths.append(str(cp))
        if paths:
            img_to_crops[str(p.resolve())] = paths
    print(f"Split hand crops ready: {len(img_to_crops)} images (newly saved={created})", flush=True)
    return img_to_crops


def load_pairs(pair_files: list[Path], img_to_crops: dict[str, list[str]]):
    rows = []
    for f in pair_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        pairs = d["pairs"] if isinstance(d, dict) and "pairs" in d else d
        for p in pairs:
            w = str(Path(p["win_image"]).resolve())
            l = str(Path(p["lose_image"]).resolve())
            wc = img_to_crops.get(w)
            lc = img_to_crops.get(l)
            if not wc or not lc:
                continue
            gid = str(p.get("group_id") or Path(w).stem.rsplit("_r", 1)[0])
            rows.append({"group": f"{f.name}::{gid}", "win": wc, "lose": lc, "src": f.name})
    return rows


def stratified_group_split(rows: list[dict], val_frac: float = 0.2, seed: int = 2027):
    rng = random.Random(seed)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r)
    train_rows, val_rows = [], []
    for src, srows in sorted(by_src.items()):
        grps = sorted({r["group"] for r in srows})
        rng.shuffle(grps)
        nv = max(1, int(round(len(grps) * val_frac)))
        vset = set(grps[:nv])
        for r in srows:
            (val_rows if r["group"] in vset else train_rows).append(r)
    return train_rows, val_rows


def aggregate_hand_scores(scores: torch.Tensor, counts: list[int], min_weight: float = 0.7) -> torch.Tensor:
    scores = scores.reshape(-1)
    out = []
    idx = 0
    for c in counts:
        s = scores[idx : idx + c]
        idx += c
        if c == 1:
            out.append(s[0])
        else:
            out.append(min_weight * s.min() + (1.0 - min_weight) * s.mean())
    return torch.stack(out)


def preprocess_all(crop_paths: list[str], processor, size: int = 384) -> dict[str, torch.Tensor]:
    tensors = {}
    for cp in crop_paths:
        img = Image.open(cp).convert("RGB")
        pv = processor(images=img, do_resize=True, size={"height": size, "width": size}, return_tensors="pt")["pixel_values"][0]
        tensors[cp] = pv.half()
    return tensors


def build_prefix_cache(
    model: HandRMv2, tensors: dict[str, torch.Tensor], device: torch.device, batch_size: int = 16
) -> tuple[dict[tuple[str, bool], torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    keys = [(cp, flp) for cp in tensors for flp in (False, True)]
    cache: dict[tuple[str, bool], torch.Tensor] = {}
    pos_emb = None
    with torch.no_grad():
        for i in range(0, len(keys), batch_size):
            chunk = keys[i : i + batch_size]
            pvs = [torch.flip(tensors[cp], dims=[-1]) if flp else tensors[cp] for cp, flp in chunk]
            pv = torch.stack(pvs).to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                hidden, pe = model.extract_prefix(pv)
            if pos_emb is None:
                pos_emb = (pe[0].detach(), pe[1].detach())
            hidden_cpu = hidden.half().cpu()
            for k, h in zip(chunk, hidden_cpu):
                cache[k] = h
    return cache, pos_emb


def eval_split(
    model: HandRMv2,
    rows: list[dict],
    prefix_cache: dict[tuple[str, bool], torch.Tensor],
    pos_emb: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 16,
):
    model.eval()
    diffs = []
    by_src = {}
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            w_list, l_list = [], []
            wc_cnt, lc_cnt = [], []
            for r in batch:
                wc_cnt.append(len(r["win"]))
                lc_cnt.append(len(r["lose"]))
                w_list.extend([prefix_cache[(p, False)] for p in r["win"]])
                l_list.extend([prefix_cache[(p, False)] for p in r["lose"]])
            all_h = torch.stack(w_list + l_list).to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                all_s = model.score_from_prefix(all_h, pos_emb)
            nw = sum(wc_cnt)
            ws = aggregate_hand_scores(all_s[:nw].float(), wc_cnt)
            ls = aggregate_hand_scores(all_s[nw:].float(), lc_cnt)
            for r, d in zip(batch, (ws - ls).cpu().tolist()):
                diffs.append(d)
                st = by_src.setdefault(r["src"], [0, 0])
                st[0] += int(d > 0)
                st[1] += 1
    acc = sum(1 for d in diffs if d > 0) / max(len(diffs), 1)
    return acc, by_src, diffs


def main():
    ap = argparse.ArgumentParser(description="Train Hand RM v2 (un-squished split hands + DINOv3 last-2-block LoRA)")
    ap.add_argument("--pairs-files", nargs="+", required=True)
    ap.add_argument("--bbox-json", default=str(DEFAULT_SD_DIR / "models" / "sam3_hand_bboxes.json"))
    ap.add_argument("--crop-dir", default=str(DEFAULT_SD_DIR / "sam3_hand_crops_v2"))
    ap.add_argument("--out-pt", default=str(DEFAULT_SD_DIR / "models" / "dinov3_hand_rm_v2.pt"))
    ap.add_argument("--dinov3-dir", default=str(DEFAULT_DINOV3_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    bbox_meta = json.loads(Path(args.bbox_json).read_text(encoding="utf-8"))
    img_to_crops = build_split_hand_crops(bbox_meta, Path(args.crop_dir))
    rows = load_pairs([Path(p) for p in args.pairs_files], img_to_crops)
    train_rows, val_rows = stratified_group_split(rows, val_frac=0.2, seed=args.seed)
    print(f"Stratified split: train={len(train_rows)} pairs, val={len(val_rows)} pairs", flush=True)

    uniq_crops = sorted({c for r in rows for c in (r["win"] + r["lose"])})
    processor = AutoImageProcessor.from_pretrained(args.dinov3_dir)
    tensors = preprocess_all(uniq_crops, processor)
    print(f"Preprocessed {len(tensors)} single-hand crops (384x384)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = HandRMv2(Path(args.dinov3_dir), lora_r=4, head_ch=16, lora_dropout=0.25).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"HandRMv2 trainable params: {n_train:,}", flush=True)

    tc = time.time()
    prefix_cache, pos_emb = build_prefix_cache(model, tensors, device)
    del tensors
    model.backbone.embeddings.cpu()
    model.backbone.rope_embeddings.cpu()
    for blk in model.backbone.model.layer[:11]:
        blk.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Cached {len(prefix_cache)} prefix tensors (layers 0..10) in {time.time()-tc:.1f}s", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    best_acc, best_state, best_ep, stale = -1.0, None, 0, 0

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        tot_loss, tot_ok, tot_n = 0.0, 0, 0
        for i in range(0, len(train_rows), args.batch_size):
            batch = train_rows[i : i + args.batch_size]
            w_list, l_list, wc_cnt, lc_cnt = [], [], [], []
            for r in batch:
                wc_cnt.append(len(r["win"]))
                lc_cnt.append(len(r["lose"]))
                w_list.extend([prefix_cache[(p, rng.random() < 0.5)] for p in r["win"]])
                l_list.extend([prefix_cache[(p, rng.random() < 0.5)] for p in r["lose"]])

            all_h = torch.stack(w_list + l_list).to(device=device, dtype=torch.float32)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                all_s = model.score_from_prefix(all_h, pos_emb)
            nw = sum(wc_cnt)
            ws = aggregate_hand_scores(all_s[:nw].float(), wc_cnt)
            ls = aggregate_hand_scores(all_s[nw:].float(), lc_cnt)
            loss = F.softplus(-(ws - ls)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()

            tot_loss += loss.item() * len(batch)
            tot_ok += int(((ws - ls) > 0).sum().item())
            tot_n += len(batch)

        val_acc, by_src, val_diffs = eval_split(model, val_rows, prefix_cache, pos_emb, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = ep
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k or k.startswith("head.")}
        else:
            stale += 1
        print(
            f"Epoch {ep:3d} ({time.time()-t0:5.1f}s) | loss={tot_loss/tot_n:.4f} | train_acc={tot_ok/tot_n:.3f} | val_acc={val_acc:.3f}",
            flush=True,
        )
        if stale >= 6 and ep >= 8:
            print(f"Early stopping at epoch {ep} (no val_acc improvement for {stale} epochs)", flush=True)
            break

    model.load_state_dict(best_state, strict=False)
    torch.save(best_state, args.out_pt)
    val_acc, by_src, val_diffs = eval_split(model, val_rows, prefix_cache, pos_emb, device)
    print(f"\nHAND_RM_V2 BEST_VAL_ACC={best_acc:.3f} at epoch {best_ep} ({len(val_rows)} val pairs) -> {args.out_pt}")
    for src, (ok, tot) in sorted(by_src.items()):
        print(f"  {src}: {ok}/{tot} = {ok/max(tot,1):.3f}")
    for thr in (0.0, 1.0, 2.0, 3.0):
        kept = [d for d in val_diffs if abs(d) >= thr]
        ok = sum(1 for d in kept if d > 0)
        print(f"  margin>={thr:.1f}: {ok}/{len(kept)} = {ok/max(len(kept),1):.3f} (kept {len(kept)}/{len(val_diffs)})")


if __name__ == "__main__":
    main()
