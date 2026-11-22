import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    DEFAULT_DINOV3_DIR,
    DEFAULT_HEAD_PT,
    Spatial2DRewardHead,
    extract_spatial24,
    run_sam3_crops,
)


def main():
    ap = argparse.ArgumentParser(description="Score rollout images with SAM3 + DINOv3 Spatial2D Hand RM and export DPO pairs")
    ap.add_argument("--img-dir", required=True, help="Directory containing generated rollout PNGs")
    ap.add_argument("--prompts-json", required=True, help="prompts_used.json matching group IDs")
    ap.add_argument("--drive-dir", required=True, help="Corresponding Google Drive directory for Colab DPO training")
    ap.add_argument("--head-pt", default=str(DEFAULT_HEAD_PT))
    ap.add_argument("--dinov3-dir", default=str(DEFAULT_DINOV3_DIR))
    ap.add_argument("--min-margin", type=float, default=2.0, help="Minimum score gap to accept a (win, lose) pair")
    ap.add_argument("--max-pairs-per-group", type=int, default=2)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    img_paths = sorted(img_dir.glob("*.png"))
    meta = run_sam3_crops(img_paths, img_dir / "sam3_hand_crops", img_dir / "sam3_hand_bboxes.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_imgs = [
        str(p.resolve())
        for p in img_paths
        if meta.get(str(p.resolve()), {}).get("crop") and Path(meta[str(p.resolve())]["crop"]).is_file()
    ]
    feats = extract_spatial24([meta[p]["crop"] for p in valid_imgs], device, Path(args.dinov3_dir))

    head = Spatial2DRewardHead(dim=768, ch=64).to(device)
    head.load_state_dict(torch.load(args.head_pt, map_location=device, weights_only=True))
    head.eval()

    scores = {}
    with torch.no_grad():
        for orig_p in valid_imgs:
            c, g = feats[meta[orig_p]["crop"]]
            scores[orig_p] = head(c.to(device).unsqueeze(0), g.to(device).unsqueeze(0)).item()

    prompts_list = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    prompt_map = {str(r["id"]): r["prompt"] for r in prompts_list}

    by_group = {}
    for orig_p, s in scores.items():
        gid = Path(orig_p).stem.rsplit("_r", 1)[0]
        by_group.setdefault(gid, []).append((s, orig_p))

    local_pairs, drive_pairs, margins = [], [], []
    for gid, items in sorted(by_group.items()):
        if len(items) < 2:
            continue
        items.sort(reverse=True)
        k_lim = min(args.max_pairs_per_group, len(items) // 2)
        for k in range(k_lim):
            sw, pw = items[k]
            sl, pl = items[-1 - k]
            margin = sw - sl
            if margin < args.min_margin:
                continue
            margins.append(margin)
            entry = {
                "group_id": gid,
                "prompt": prompt_map.get(gid, ""),
                "win_image": pw,
                "lose_image": pl,
                "win_id": Path(pw).stem,
                "lose_id": Path(pl).stem,
                "score_win": round(sw, 4),
                "score_lose": round(sl, 4),
                "margin": round(margin, 4),
            }
            local_pairs.append(entry)
            dq = dict(entry)
            dq["win_image"] = args.drive_dir.rstrip("/") + "/" + Path(pw).name
            dq["lose_image"] = args.drive_dir.rstrip("/") + "/" + Path(pl).name
            drive_pairs.append(dq)

    out_local = img_dir / "dpo_pairs_auto_local.json"
    out_drive = img_dir / "dpo_pairs_auto_drive.json"
    out_local.write_text(json.dumps({"count": len(local_pairs), "pairs": local_pairs}, indent=2, ensure_ascii=False), encoding="utf-8")
    out_drive.write_text(json.dumps({"count": len(drive_pairs), "pairs": drive_pairs}, indent=2, ensure_ascii=False), encoding="utf-8")
    avg_m = sum(margins) / max(len(margins), 1)
    print(f"AUTO_DPO_DONE: {len(local_pairs)} pairs (min_margin={args.min_margin}, avg_margin={avg_m:.2f}) -> {out_local}", flush=True)


if __name__ == "__main__":
    main()
