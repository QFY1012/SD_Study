import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import DEFAULT_SD_DIR
from train_hand_rm_v2 import build_split_hand_crops, load_pairs
from train_hand_rm_v3_5fold import make_stratified_group_folds


class PairwiseCrossHandRM(nn.Module):
    """Antisymmetric Pairwise Hand RM using Spatial Cross-Attention + Difference Pooling."""

    def __init__(self, dim: int = 768, ch: int = 16, dropout: float = 0.30):
        super().__init__()
        self.proj = nn.Conv2d(dim, ch, kernel_size=1)
        self.norm = nn.GroupNorm(1, ch)
        self.local_ctx = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.cls_proj = nn.Linear(dim, ch)
        self.scale = ch ** -0.5

        self.q_proj = nn.Linear(ch, ch)
        self.k_proj = nn.Linear(ch, ch)
        self.v_proj = nn.Linear(ch, ch)

        # Features from directed diff (A vs aligned B): avg(diff), max(diff), min(diff), cls_diff
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch * 4, ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ch, 1),
        )

    def encode_hand(self, cls_v: torch.Tensor, grid: torch.Tensor):
        f = F.gelu(self.norm(self.proj(grid)))
        f = f + F.gelu(self.local_ctx(f))
        c = F.gelu(self.cls_proj(cls_v))
        return c, f  # [B, ch], [B, ch, 24, 24]

    def directed_score(self, c_a, f_a, c_b, f_b) -> torch.Tensor:
        b, ch, h, w = f_a.shape
        ta = f_a.flatten(2).transpose(1, 2)  # [B, 576, ch]
        tb = f_b.flatten(2).transpose(1, 2)  # [B, 576, ch]

        q = self.q_proj(ta)
        k = self.k_proj(tb)
        v = self.v_proj(tb)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        tb_aligned = torch.bmm(attn, v)

        diff = ta - tb_aligned  # [B, 576, ch]
        d_avg = diff.mean(dim=1)
        d_max = diff.max(dim=1).values
        d_min = diff.min(dim=1).values
        c_diff = c_a - c_b
        feat = torch.cat([d_avg, d_max, d_min, c_diff], dim=-1)
        return self.head(feat).squeeze(-1)

    def forward_pair(self, cls_a, grid_a, cls_b, grid_b) -> torch.Tensor:
        c_a, f_a = self.encode_hand(cls_a, grid_a)
        c_b, f_b = self.encode_hand(cls_b, grid_b)
        # Strictly antisymmetric: logit(A, B) == -logit(B, A)
        return self.directed_score(c_a, f_a, c_b, f_b) - self.directed_score(c_b, f_b, c_a, f_a)


def build_matched_hands(batch: list[dict], feats: dict, flip_prob: float, rng: random.Random):
    cls_a, grid_a, cls_b, grid_b, counts = [], [], [], [], []
    for r in batch:
        wh, lh = r["win"], r["lose"]
        n_match = min(len(wh), len(lh))
        counts.append(n_match)
        flp_w = rng.random() < flip_prob if flip_prob > 0 else False
        flp_l = rng.random() < flip_prob if flip_prob > 0 else False
        for hi in range(n_match):
            ca, ga = feats[(wh[hi], flp_w)]
            cb, gb = feats[(lh[hi], flp_l)]
            cls_a.append(ca)
            grid_a.append(ga)
            cls_b.append(cb)
            grid_b.append(gb)
    return cls_a, grid_a, cls_b, grid_b, counts


def aggregate_pair_logits(hand_logits: torch.Tensor, counts: list[int]) -> torch.Tensor:
    out, idx = [], 0
    for c in counts:
        s = hand_logits[idx : idx + c]
        idx += c
        out.append(s.mean())
    return torch.stack(out)


def eval_fold(model: PairwiseCrossHandRM, rows: list[dict], feats: dict, device: torch.device, batch_size: int = 32):
    model.eval()
    logits_all = []
    tot_loss = 0.0
    rng = random.Random(0)
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            ca, ga, cb, gb, counts = build_matched_hands(batch, feats, 0.0, rng)
            hl = model.forward_pair(
                torch.stack(ca).to(device=device, dtype=torch.float32),
                torch.stack(ga).to(device=device, dtype=torch.float32),
                torch.stack(cb).to(device=device, dtype=torch.float32),
                torch.stack(gb).to(device=device, dtype=torch.float32),
            )
            pl = aggregate_pair_logits(hl, counts)
            tot_loss += F.softplus(-pl).sum().item()
            logits_all.extend(pl.cpu().tolist())
    acc = sum(1 for d in logits_all if d > 0) / max(len(logits_all), 1)
    return acc, tot_loss / max(len(logits_all), 1), logits_all


def main():
    ap = argparse.ArgumentParser(description="Train Pairwise Cross-Attention Hand RM v4")
    ap.add_argument("--pairs-files", nargs="+", required=True)
    ap.add_argument("--bbox-json", default=str(DEFAULT_SD_DIR / "models" / "sam3_hand_bboxes.json"))
    ap.add_argument("--crop-dir", default=str(DEFAULT_SD_DIR / "sam3_hand_crops_v2"))
    ap.add_argument("--feat-cache", default=str(DEFAULT_SD_DIR / "models" / "dinov3_single_hand_feats_v3.pt"))
    ap.add_argument("--out-pt", default=str(DEFAULT_SD_DIR / "models" / "hand_rm_v4_pairwise_5fold.pt"))
    ap.add_argument("--oof-json", default=str(DEFAULT_SD_DIR / "models" / "hand_rm_v4_oof_audit.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    bbox_meta = json.loads(Path(args.bbox_json).read_text(encoding="utf-8"))
    img_to_crops = build_split_hand_crops(bbox_meta, Path(args.crop_dir))
    rows = load_pairs([Path(p) for p in args.pairs_files], img_to_crops)
    print(f"Loaded {len(rows)} pairs", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.load(args.feat_cache, map_location="cpu")
    if device.type == "cuda":
        feats = {k: (v[0].to(device), v[1].to(device)) for k, v in feats.items()}
        print(f"Preloaded {len(feats)} cached tensors onto GPU", flush=True)

    probe = PairwiseCrossHandRM(dim=768, ch=args.ch)
    print(f"PairwiseCrossHandRM params per fold: {sum(p.numel() for p in probe.parameters()):,}", flush=True)

    folds = make_stratified_group_folds(rows, n_folds=args.n_folds, seed=args.seed)
    fold_states, oof_records = [], []
    t0 = time.time()

    for k in range(args.n_folds):
        val_rows = folds[k]
        train_rows = [r for fi, f in enumerate(folds) if fi != k for r in f]
        torch.manual_seed(args.seed + k)
        rng = random.Random(args.seed + k)
        model = PairwiseCrossHandRM(dim=768, ch=args.ch, dropout=0.30).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc, best_loss, best_ep, best_state, best_logits = -1.0, 1e9, 0, None, []
        tr = list(train_rows)
        for ep in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(tr)
            for i in range(0, len(tr), args.batch_size):
                batch = tr[i : i + args.batch_size]
                ca, ga, cb, gb, counts = build_matched_hands(batch, feats, 0.5, rng)
                opt.zero_grad()
                hl = model.forward_pair(
                    torch.stack(ca).to(device=device, dtype=torch.float32),
                    torch.stack(ga).to(device=device, dtype=torch.float32),
                    torch.stack(cb).to(device=device, dtype=torch.float32),
                    torch.stack(gb).to(device=device, dtype=torch.float32),
                )
                pl = aggregate_pair_logits(hl, counts)
                loss = F.softplus(-pl).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            va, vl, logits = eval_fold(model, val_rows, feats, device)
            if va > best_acc or (va == best_acc and vl < best_loss):
                best_acc, best_loss, best_ep = va, vl, ep
                best_logits = logits
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

        fold_states.append(best_state)
        print(f"Fold {k+1}/{args.n_folds} | train={len(train_rows)} val={len(val_rows)} | best_val_acc={best_acc:.3f} (ep {best_ep}, loss={best_loss:.4f})", flush=True)
        for r, d in zip(val_rows, best_logits):
            oof_records.append({"fold": k + 1, "src": r["src"], "group": r["group"], "win": r["win"], "lose": r["lose"], "diff": d})

    oof_diffs = [x["diff"] for x in oof_records]
    oof_acc = sum(1 for d in oof_diffs if d > 0) / max(len(oof_diffs), 1)
    print(f"\nHAND_RM_V4_PAIRWISE OOF_ACC={oof_acc:.3f} ({sum(d>0 for d in oof_diffs)}/{len(oof_diffs)}) in {time.time()-t0:.1f}s", flush=True)

    by_src = {}
    for x in oof_records:
        st = by_src.setdefault(x["src"], [0, 0])
        st[0] += int(x["diff"] > 0)
        st[1] += 1
    for src, (ok, tot) in sorted(by_src.items()):
        print(f"  {src}: {ok}/{tot} = {ok/max(tot,1):.3f}")

    abs_sorted = sorted([abs(d) for d in oof_diffs], reverse=True)
    for pct in (75, 50, 30, 20):
        thr = abs_sorted[min(len(abs_sorted) - 1, int(len(abs_sorted) * pct / 100) - 1)]
        kept = [d for d in oof_diffs if abs(d) >= thr]
        ok = sum(1 for d in kept if d > 0)
        print(f"  top {pct:2d}% (|diff|>={thr:.3f}): {ok}/{len(kept)} = {ok/max(len(kept),1)*100:.1f}%")

    torch.save({"ch": args.ch, "n_folds": args.n_folds, "folds": fold_states}, args.out_pt)
    Path(args.oof_json).write_text(json.dumps(sorted(oof_records, key=lambda x: x["diff"]), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Pairwise 5-fold -> {args.out_pt}")


if __name__ == "__main__":
    main()
