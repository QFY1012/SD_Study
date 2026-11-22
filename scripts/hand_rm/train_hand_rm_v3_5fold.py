import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import DEFAULT_DINOV3_DIR, DEFAULT_SD_DIR
from train_hand_rm_v2 import aggregate_hand_scores, build_split_hand_crops, load_pairs


class GatedSpatialRewardHead(nn.Module):
    """Lightweight gated-attention spatial pooling head over frozen DINOv3 features."""

    def __init__(self, dim: int = 768, ch: int = 16, dropout: float = 0.30):
        super().__init__()
        self.patch_proj = nn.Conv2d(dim, ch, kernel_size=1)
        self.local_ctx = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, ch)
        # Gated attention over 24x24 patch tokens (focuses on fingertips / finger gaps)
        self.attn_v = nn.Conv2d(ch, ch, kernel_size=1)
        self.attn_u = nn.Conv2d(ch, ch, kernel_size=1)
        self.attn_w = nn.Conv2d(ch, 1, kernel_size=1)

        self.cls_proj = nn.Linear(dim, ch)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(ch * 2, 1)

    def forward(self, cls_vec: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm(self.patch_proj(grid)))
        h = h + F.gelu(self.local_ctx(h))

        a = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        logits = self.attn_w(a).flatten(2)  # [B, 1, 576]
        weights = F.softmax(logits, dim=-1)
        attn_pool = torch.bmm(weights, h.flatten(2).transpose(1, 2)).squeeze(1)  # [B, ch]

        c = F.gelu(self.cls_proj(cls_vec))
        feat = self.drop(torch.cat([c, attn_pool], dim=-1))
        return self.out(feat).squeeze(-1)


def extract_frozen_crops(
    crop_paths: list[str], dinov3_dir: Path, device: torch.device, batch_size: int = 16
) -> dict[tuple[str, bool], tuple[torch.Tensor, torch.Tensor]]:
    """Extract normalized CLS [768] and 24x24 patch grid [768, 24, 24] for orig + hflip."""
    processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModel.from_pretrained(str(dinov3_dir), dtype=dtype).to(device).eval()
    n_reg = getattr(backbone.config, "num_register_tokens", 4)

    feats: dict[tuple[str, bool], tuple[torch.Tensor, torch.Tensor]] = {}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(crop_paths), batch_size):
            chunk = crop_paths[i : i + batch_size]
            imgs_orig = [Image.open(p).convert("RGB") for p in chunk]
            imgs_flip = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in imgs_orig]
            for is_flip, imgs in ((False, imgs_orig), (True, imgs_flip)):
                inputs = processor(images=imgs, do_resize=True, size={"height": 384, "width": 384}, return_tensors="pt")
                inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
                tokens = backbone(**inputs).last_hidden_state
                cls_v = F.normalize(tokens[:, 0, :].float(), dim=-1).half().cpu()
                patches = F.normalize(tokens[:, 1 + n_reg :, :].float(), dim=-1)
                side = int(patches.shape[1] ** 0.5)
                grid_v = patches.transpose(1, 2).reshape(len(chunk), 768, side, side).half().cpu()
                for idx, p in enumerate(chunk):
                    feats[(p, is_flip)] = (cls_v[idx], grid_v[idx])
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Extracted frozen DINOv3 features for {len(crop_paths)}x2 crops in {time.time()-t0:.1f}s", flush=True)
    return feats


def make_stratified_group_folds(rows: list[dict], n_folds: int = 5, seed: int = 2027) -> list[list[dict]]:
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r)

    folds: list[list[dict]] = [[] for _ in range(n_folds)]
    for src, srows in sorted(by_src.items()):
        grps: dict[str, list[dict]] = {}
        for r in srows:
            grps.setdefault(r["group"], []).append(r)
        grp_keys = sorted(grps.keys())
        rng.shuffle(grp_keys)
        for idx, gk in enumerate(grp_keys):
            folds[idx % n_folds].extend(grps[gk])
    return folds


def eval_fold(head: GatedSpatialRewardHead, rows: list[dict], feats: dict, device: torch.device, batch_size: int = 32):
    head.eval()
    diffs = []
    tot_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            wc, lc, cls_l, grid_l = [], [], [], []
            for r in batch:
                wc.append(len(r["win"]))
                lc.append(len(r["lose"]))
                for p in r["win"]:
                    c, g = feats[(p, False)]
                    cls_l.append(c)
                    grid_l.append(g)
                for p in r["lose"]:
                    c, g = feats[(p, False)]
                    cls_l.append(c)
                    grid_l.append(g)
            cls_t = torch.stack(cls_l).to(device=device, dtype=torch.float32)
            grid_t = torch.stack(grid_l).to(device=device, dtype=torch.float32)
            all_s = head(cls_t, grid_t)
            nw = sum(wc)
            ws = aggregate_hand_scores(all_s[:nw], wc)
            ls = aggregate_hand_scores(all_s[nw:], lc)
            d = ws - ls
            tot_loss += F.softplus(-d).sum().item()
            diffs.extend(d.cpu().tolist())
    acc = sum(1 for d in diffs if d > 0) / max(len(diffs), 1)
    return acc, tot_loss / max(len(diffs), 1), diffs


def train_one_fold(
    fold_id: int,
    train_rows: list[dict],
    val_rows: list[dict],
    feats: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    ch: int,
    seed: int,
):
    torch.manual_seed(seed + fold_id)
    rng = random.Random(seed + fold_id)
    head = GatedSpatialRewardHead(dim=768, ch=ch, dropout=0.30).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_loss, best_ep, best_state, best_diffs = -1.0, 1e9, 0, None, []
    rows = list(train_rows)
    for ep in range(1, epochs + 1):
        head.train()
        rng.shuffle(rows)
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            wc, lc, cls_l, grid_l = [], [], [], []
            for r in batch:
                wc.append(len(r["win"]))
                lc.append(len(r["lose"]))
                for p in r["win"]:
                    c, g = feats[(p, rng.random() < 0.5)]
                    cls_l.append(c)
                    grid_l.append(g)
                for p in r["lose"]:
                    c, g = feats[(p, rng.random() < 0.5)]
                    cls_l.append(c)
                    grid_l.append(g)
            cls_t = torch.stack(cls_l).to(device=device, dtype=torch.float32)
            grid_t = torch.stack(grid_l).to(device=device, dtype=torch.float32)
            opt.zero_grad()
            all_s = head(cls_t, grid_t)
            nw = sum(wc)
            ws = aggregate_hand_scores(all_s[:nw], wc)
            ls = aggregate_hand_scores(all_s[nw:], lc)
            loss = F.softplus(-(ws - ls)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

        va, vl, diffs = eval_fold(head, val_rows, feats, device)
        if va > best_acc or (va == best_acc and vl < best_loss):
            best_acc, best_loss, best_ep = va, vl, ep
            best_diffs = diffs
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    return best_acc, best_loss, best_ep, best_state, best_diffs


def main():
    ap = argparse.ArgumentParser(description="Train Hand RM v3 (un-squished split hands + GatedAttention + 5-fold ensemble)")
    ap.add_argument("--pairs-files", nargs="+", required=True)
    ap.add_argument("--bbox-json", default=str(DEFAULT_SD_DIR / "models" / "sam3_hand_bboxes.json"))
    ap.add_argument("--crop-dir", default=str(DEFAULT_SD_DIR / "sam3_hand_crops_v2"))
    ap.add_argument("--feat-cache", default=str(DEFAULT_SD_DIR / "models" / "dinov3_single_hand_feats_v3.pt"))
    ap.add_argument("--out-pt", default=str(DEFAULT_SD_DIR / "models" / "hand_rm_v3_5fold.pt"))
    ap.add_argument("--oof-json", default=str(DEFAULT_SD_DIR / "models" / "hand_rm_v3_oof_audit.json"))
    ap.add_argument("--dinov3-dir", default=str(DEFAULT_DINOV3_DIR))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.10)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    bbox_meta = json.loads(Path(args.bbox_json).read_text(encoding="utf-8"))
    img_to_crops = build_split_hand_crops(bbox_meta, Path(args.crop_dir))
    rows = load_pairs([Path(p) for p in args.pairs_files], img_to_crops)
    uniq_crops = sorted({c for r in rows for c in (r["win"] + r["lose"])})
    print(f"Loaded {len(rows)} pairs ({len(uniq_crops)} unique single-hand crops)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_p = Path(args.feat_cache)
    feats = torch.load(cache_p, map_location="cpu") if cache_p.is_file() else {}
    missing = [p for p in uniq_crops if (p, False) not in feats or (p, True) not in feats]
    if missing:
        new_feats = extract_frozen_crops(missing, Path(args.dinov3_dir), device)
        feats.update(new_feats)
        torch.save(feats, cache_p)
        print(f"Saved feature cache -> {cache_p} ({len(feats)//2} crops)", flush=True)

    if device.type == "cuda":
        feats = {k: (v[0].to(device), v[1].to(device)) for k, v in feats.items()}
        print(f"Preloaded {len(feats)} cached feature tensors onto GPU", flush=True)

    probe = GatedSpatialRewardHead(dim=768, ch=args.ch)
    print(f"GatedSpatialRewardHead params per fold: {sum(p.numel() for p in probe.parameters()):,}", flush=True)

    folds = make_stratified_group_folds(rows, n_folds=args.n_folds, seed=args.seed)
    fold_states = []
    oof_records = []
    t0 = time.time()
    for k in range(args.n_folds):
        val_rows = folds[k]
        train_rows = [r for fi, f in enumerate(folds) if fi != k for r in f]
        ba, bl, be, st, diffs = train_one_fold(
            k, train_rows, val_rows, feats, device, args.epochs, args.batch_size, args.lr, args.weight_decay, args.ch, args.seed
        )
        fold_states.append(st)
        print(f"Fold {k+1}/{args.n_folds} | train={len(train_rows)} val={len(val_rows)} | best_val_acc={ba:.3f} (ep {be}, loss={bl:.4f})", flush=True)
        for r, d in zip(val_rows, diffs):
            oof_records.append({"fold": k + 1, "src": r["src"], "group": r["group"], "win": r["win"], "lose": r["lose"], "diff": d})

    oof_diffs = [x["diff"] for x in oof_records]
    oof_acc = sum(1 for d in oof_diffs if d > 0) / max(len(oof_diffs), 1)
    print(f"\nHAND_RM_V3_5FOLD OOF_ACC={oof_acc:.3f} ({sum(d>0 for d in oof_diffs)}/{len(oof_diffs)}) in {time.time()-t0:.1f}s", flush=True)

    by_src: dict[str, list[int]] = {}
    for x in oof_records:
        st = by_src.setdefault(x["src"], [0, 0])
        st[0] += int(x["diff"] > 0)
        st[1] += 1
    for src, (ok, tot) in sorted(by_src.items()):
        print(f"  {src}: {ok}/{tot} = {ok/max(tot,1):.3f}")

    for thr in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        kept = [d for d in oof_diffs if abs(d) >= thr]
        ok = sum(1 for d in kept if d > 0)
        print(f"  margin>={thr:.1f}: recall={len(kept)}/{len(oof_diffs)} ({len(kept)/len(oof_diffs)*100:.1f}%), acc={ok}/{len(kept)} ({ok/max(len(kept),1)*100:.1f}%)")

    abs_sorted = sorted([abs(d) for d in oof_diffs], reverse=True)
    for pct in (75, 50, 30, 20):
        thr = abs_sorted[min(len(abs_sorted) - 1, int(len(abs_sorted) * pct / 100) - 1)]
        kept = [d for d in oof_diffs if abs(d) >= thr]
        ok = sum(1 for d in kept if d > 0)
        print(f"  top {pct:2d}% (|diff|>={thr:.3f}): {ok}/{len(kept)} = {ok/max(len(kept),1)*100:.1f}%")

    torch.save({"ch": args.ch, "n_folds": args.n_folds, "folds": fold_states}, args.out_pt)
    oof_sorted = sorted(oof_records, key=lambda x: x["diff"])
    Path(args.oof_json).write_text(json.dumps(oof_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved 5-fold ensemble -> {args.out_pt}\nSaved OOF audit -> {args.oof_json}")


if __name__ == "__main__":
    main()
