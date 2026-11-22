import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    DEFAULT_DINOV3_DIR,
    DEFAULT_HEAD_PT,
    DEFAULT_SD_DIR,
    Spatial2DRewardHead,
    extract_spatial24,
    run_sam3_crops,
)


def load_pairs(pair_files: list[Path], sd_dir: Path, bbox_meta: dict):
    rows = []
    for f in pair_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        pairs = d["pairs"] if isinstance(d, dict) and "pairs" in d else d
        for p in pairs:
            w = Path(p["win_image"])
            l = Path(p["lose_image"])
            if not w.is_absolute():
                w = sd_dir / w
            if not l.is_absolute():
                l = sd_dir / l
            wc = bbox_meta.get(str(w.resolve()), {}).get("crop")
            lc = bbox_meta.get(str(l.resolve()), {}).get("crop")
            if not wc or not lc or not Path(wc).is_file() or not Path(lc).is_file():
                continue
            gid = str(p.get("group_id") or w.stem.rsplit("_r", 1)[0])
            rows.append({"group": gid, "win": wc, "lose": lc, "src": f.name})
    return rows


def eval_split(head, pairs, feats, device):
    head.eval()
    diffs = []
    by_src = {}
    with torch.no_grad():
        for r in pairs:
            cw, gw = feats[r["win"]]
            cl, gl = feats[r["lose"]]
            rw = head(cw.to(device).unsqueeze(0), gw.to(device).unsqueeze(0)).item()
            rl = head(cl.to(device).unsqueeze(0), gl.to(device).unsqueeze(0)).item()
            diff = rw - rl
            diffs.append(diff)
            s = by_src.setdefault(r["src"], [0, 0])
            s[0] += int(diff > 0)
            s[1] += 1
    acc = sum(1 for d in diffs if d > 0) / max(len(diffs), 1)
    return acc, by_src, diffs


def main():
    ap = argparse.ArgumentParser(description="Train DINOv3 Spatial2D Hand Reward Head from SAM3 crops")
    ap.add_argument("--pairs-files", nargs="+", required=True)
    ap.add_argument("--sd-dir", default=str(DEFAULT_SD_DIR))
    ap.add_argument("--crop-dir", default=str(DEFAULT_SD_DIR / "sam3_hand_crops"))
    ap.add_argument("--bbox-json", default=str(DEFAULT_SD_DIR / "models" / "sam3_hand_bboxes.json"))
    ap.add_argument("--cache-pt", default=str(DEFAULT_SD_DIR / "models" / "dinov3_hand_crop_spatial24.pt"))
    ap.add_argument("--out-head", default=str(DEFAULT_HEAD_PT))
    ap.add_argument("--dinov3-dir", default=str(DEFAULT_DINOV3_DIR))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    sd_dir = Path(args.sd_dir)
    pair_files = [Path(p) for p in args.pairs_files]

    all_imgs = set()
    for f in pair_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for p in (d["pairs"] if isinstance(d, dict) and "pairs" in d else d):
            for k in ("win_image", "lose_image"):
                pth = Path(p[k])
                if not pth.is_absolute():
                    pth = sd_dir / pth
                if pth.is_file():
                    all_imgs.add(pth)

    bbox_meta = run_sam3_crops(sorted(all_imgs), Path(args.crop_dir), Path(args.bbox_json), prefix_parent=True)
    rows = load_pairs(pair_files, sd_dir, bbox_meta)
    uniq_crops = sorted({r["win"] for r in rows} | {r["lose"] for r in rows})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_pt = Path(args.cache_pt)
    if cache_pt.is_file():
        feats = torch.load(str(cache_pt), weights_only=True)
        missing = [p for p in uniq_crops if p not in feats]
        if missing:
            feats.update(extract_spatial24(missing, device, Path(args.dinov3_dir)))
            torch.save(feats, str(cache_pt))
    else:
        feats = extract_spatial24(uniq_crops, device, Path(args.dinov3_dir))
        cache_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, str(cache_pt))

    rng = random.Random(args.seed)
    groups = sorted({r["group"] for r in rows})
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * 0.2))
    val_groups = set(groups[:n_val])

    train_rows = [r for r in rows if r["group"] not in val_groups]
    val_rows = [r for r in rows if r["group"] in val_groups]
    print(f"Split: train={len(train_rows)} pairs ({len(groups)-n_val} groups), val={len(val_rows)} pairs ({n_val} groups)", flush=True)

    torch.manual_seed(args.seed)
    head = Spatial2DRewardHead(dim=768, ch=64).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-2)

    cw_tr = torch.stack([feats[r["win"]][0] for r in train_rows])
    gw_tr = torch.stack([feats[r["win"]][1] for r in train_rows])
    cl_tr = torch.stack([feats[r["lose"]][0] for r in train_rows])
    gl_tr = torch.stack([feats[r["lose"]][1] for r in train_rows])

    best_val, best_by_src, best_diffs = 0.0, {}, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(train_rows))
        tot_loss, ok_tr = 0.0, 0
        for i in range(0, len(train_rows), args.batch_size):
            idx = perm[i : i + args.batch_size]
            rw = head(cw_tr[idx].to(device), gw_tr[idx].to(device)).squeeze(-1)
            rl = head(cl_tr[idx].to(device), gl_tr[idx].to(device)).squeeze(-1)
            loss = -F.logsigmoid(rw - rl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            ok_tr += (rw > rl).sum().item()

        if epoch % 15 == 0 or epoch == 1 or epoch == args.epochs:
            val_acc, by_src, diffs = eval_split(head, val_rows, feats, device)
            if val_acc >= best_val:
                best_val, best_by_src, best_diffs = val_acc, by_src, diffs
                Path(args.out_head).parent.mkdir(parents=True, exist_ok=True)
                torch.save(head.state_dict(), args.out_head)
            print(f"Epoch {epoch:3d} | loss={tot_loss/len(train_rows):.4f} | train_acc={ok_tr/len(train_rows):.3f} | val_acc={val_acc:.3f}", flush=True)

    print(f"\nBEST_VAL_ACC={best_val:.3f} ({len(val_rows)} val pairs) -> {args.out_head}", flush=True)
    for src, (c, t) in sorted(best_by_src.items()):
        print(f"  {src}: {c}/{t} = {c/max(t,1):.3f}", flush=True)
    for thr in (0.0, 1.0, 2.0, 3.0):
        sub = [d for d in best_diffs if abs(d) >= thr]
        ok = sum(1 for d in sub if d > 0)
        print(f"  margin>={thr:.1f}: {ok}/{len(sub)} = {ok/max(len(sub),1):.3f} (kept {len(sub)}/{len(best_diffs)})", flush=True)


if __name__ == "__main__":
    main()
