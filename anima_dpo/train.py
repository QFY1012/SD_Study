"""Offline Flow-DPO LoRA trainer for Anima. Invoke: python -m anima_dpo.train --config ..."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

from .loss import dpo_loss, flow_noisy_latent, per_sample_mse, sample_flow_sigmas


def load_pairs(pairs_path: str | Path) -> list[dict[str, str]]:
    """Load and validate human/curated (prompt, win_image, lose_image) DPO pairs."""
    raw = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    items = raw["pairs"] if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError("pairs_json must contain a non-empty list of pairs (or {'pairs': [...]})")
    cleaned: list[dict[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Pair #{idx} must be an object")
        prompt = str(item.get("prompt", "")).strip()
        win = str(item.get("win_image", "")).strip()
        lose = str(item.get("lose_image", "")).strip()
        if not prompt or not win or not lose:
            raise ValueError(f"Pair #{idx} must have nonempty prompt, win_image, and lose_image")
        if win == lose:
            raise ValueError(f"Pair #{idx} has identical win_image and lose_image")
        if not Path(win).is_file():
            raise FileNotFoundError(f"Missing win_image for pair #{idx}: {win}")
        if not Path(lose).is_file():
            raise FileNotFoundError(f"Missing lose_image for pair #{idx}: {lose}")
        cleaned.append({"prompt": prompt, "win_image": win, "lose_image": lose})
    return cleaned


@contextmanager
def temporary_param_override(params: list[Any], ref_tensors: list[Any]):
    """Temporarily replace trainable LoRA parameter values with frozen ref values inside no_grad."""
    import torch

    backups = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for p, r in zip(params, ref_tensors):
                p.copy_(r.to(device=p.device, dtype=p.dtype))
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.copy_(b)


def load_image_tensor(path: str | Path, width: int, height: int):
    """Load RGB image normalized to [-1, 1] tensor of shape [1, 3, H, W]."""
    import numpy as np
    from PIL import Image
    import torch

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), resampling.LANCZOS)
        arr = np.asarray(rgb, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def validate_config(config: dict[str, Any]) -> None:
    required = ["sd_scripts", "dit", "text_encoder", "vae", "lora", "pairs_file", "output"]
    for key in required:
        if key not in config:
            raise ValueError("Missing config key: " + key)
    width, height = config.get("width", 768), config.get("height", 1024)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if config.get("epochs", 2) < 1 or config.get("grad_accum", 4) < 1 or config.get("batch_size", 1) < 1:
        raise ValueError("epochs, grad_accum, and batch_size must be >= 1")
    if config.get("beta_dpo", 2000.0) <= 0 or config.get("learning_rate", 2e-6) <= 0:
        raise ValueError("beta_dpo and learning_rate must be positive")
    if config.get("flow_shift", 5.0) <= 0:
        raise ValueError("flow_shift must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true", help="Validate config & pairs without loading GPU weights.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_config:
        pairs_info = len(load_pairs(config["pairs_file"])) if Path(config["pairs_file"]).is_file() else "deferred (remote path)"
        print(f"DPO config valid (pairs: {pairs_info}); weights and CUDA not tested")
        return
    pairs = load_pairs(config["pairs_file"])

    sys.path.insert(0, str(Path(config["sd_scripts"]).resolve()))
    import gc
    import torch
    from anima_grpo.backend import AnimaBackend

    if not torch.cuda.is_available():
        raise RuntimeError("Anima DPO training requires CUDA")

    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    seed = int(config.get("seed", 42))
    width = int(config.get("width", 768))
    height = int(config.get("height", 1024))
    epochs = int(config.get("epochs", 2))
    batch_size = int(config.get("batch_size", 1))
    grad_accum = int(config.get("grad_accum", 4))
    beta_dpo = float(config.get("beta_dpo", 2000.0))
    lr = float(config.get("learning_rate", 2e-6))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    flow_shift = float(config.get("flow_shift", 5.0))
    sigma_scheme = str(config.get("sigma_scheme", "logit_normal"))
    save_every_steps = int(config.get("save_every_steps", 25))

    torch.manual_seed(seed)
    rng = random.Random(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)

    backend = AnimaBackend(config, torch.device("cuda"))
    if config.get("gradient_checkpointing", batch_size > 1) and hasattr(backend.model, "enable_gradient_checkpointing"):
        backend.model.enable_gradient_checkpointing()
        print("ENABLED_GRADIENT_CHECKPOINTING", flush=True)
    unique_prompts = list(dict.fromkeys(p["prompt"] for p in pairs))
    backend.cache_prompts(unique_prompts, config.get("negative_prompt", ""))

    # Encode all unique win/lose images once with QwenImage 2D VAE, then release VAE from VRAM/RAM.
    unique_imgs = list(dict.fromkeys(img for p in pairs for img in (p["win_image"], p["lose_image"])))
    latent_cache: dict[str, torch.Tensor] = {}
    backend.vae.to(backend.device)
    with torch.no_grad():
        for img_path in unique_imgs:
            px = load_image_tensor(img_path, width, height).to(backend.device, dtype=torch.bfloat16)
            lat = backend.vae.encode_pixels_to_latents(px).unsqueeze(2).cpu()  # [1, 16, 1, H/8, W/8]
            latent_cache[img_path] = lat
    backend.vae.to("cpu")
    del backend.vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"ENCODED_LATENTS: {len(latent_cache)} images cached on CPU", flush=True)

    freeze_blocks = set(int(b) for b in config.get("freeze_blocks", []))
    if freeze_blocks:
        frozen_cnt = 0
        for name, p in backend.network.named_parameters():
            m = re.search(r"blocks_(\d+)_", name)
            if m and int(m.group(1)) in freeze_blocks:
                p.requires_grad_(False)
                frozen_cnt += 1
        backend.params = [p for p in backend.network.parameters() if p.requires_grad]
        print(f"FROZEN_BLOCKS: {sorted(freeze_blocks)} ({frozen_cnt} tensors frozen, {len(backend.params)} trainable)", flush=True)

    # Keep frozen reference snapshot of initial LoRA weights on CPU (~66MB).
    if config.get("ref_lora"):
        from networks import lora_anima
        _, ref_state = lora_anima.create_network_from_weights(
            1.0, config["ref_lora"], None, [backend.text_encoder], backend.model, for_inference=False
        )
        ref_tensors = [
            ref_state[name].detach().cpu().to(dtype=p.dtype).clone()
            for name, p in backend.network.named_parameters()
            if p.requires_grad
        ]
        del ref_state
        print(f"LOADED_REF_LORA: {config['ref_lora']} ({len(ref_tensors)} tensors)", flush=True)
    else:
        ref_tensors = [p.detach().cpu().clone() for p in backend.params]
    optimizer = torch.optim.AdamW(backend.params, lr=lr, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    accum_count = 0
    accum_stats: list[dict[str, float]] = []
    history: list[dict[str, Any]] = []
    t_start = time.time()

    for epoch in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for start_idx in range(0, len(order), batch_size):
            batch_indices = order[start_idx : start_idx + batch_size]
            bsz = len(batch_indices)
            batch_prompts = [pairs[i]["prompt"] for i in batch_indices]
            lat_w = torch.cat([latent_cache[pairs[i]["win_image"]] for i in batch_indices], dim=0).to(backend.device)
            lat_l = torch.cat([latent_cache[pairs[i]["lose_image"]] for i in batch_indices], dim=0).to(backend.device)
            noise = torch.randn(lat_w.shape, generator=gen, device=backend.device, dtype=torch.float32)
            sigma = sample_flow_sigmas(bsz, scheme=sigma_scheme, flow_shift=flow_shift, device=backend.device, generator=gen)

            xt_w, target_w = flow_noisy_latent(lat_w, noise, sigma)
            xt_l, target_l = flow_noisy_latent(lat_l, noise, sigma)
            xt_both = torch.cat([xt_w, xt_l], dim=0)
            sigmas_both = torch.cat([sigma, sigma], dim=0)
            prompts_both = batch_prompts + batch_prompts

            with torch.no_grad(), temporary_param_override(backend.params, ref_tensors):
                v_ref = backend.velocity_batch(xt_both, sigmas_both, prompts_both)
                v_ref_w, v_ref_l = v_ref.chunk(2, dim=0)
                mse_ref_w = per_sample_mse(v_ref_w, target_w)
                mse_ref_l = per_sample_mse(v_ref_l, target_l)

            v_pol = backend.velocity_batch(xt_both, sigmas_both, prompts_both)
            v_pol_w, v_pol_l = v_pol.chunk(2, dim=0)
            mse_pol_w = per_sample_mse(v_pol_w, target_w)
            mse_pol_l = per_sample_mse(v_pol_l, target_l)

            loss, step_stats = dpo_loss(mse_pol_w, mse_pol_l, mse_ref_w, mse_ref_l, beta=beta_dpo)
            (loss / grad_accum).backward()
            accum_count += 1
            accum_stats.append(step_stats)

            is_last_batch = epoch == epochs - 1 and start_idx + batch_size >= len(order)
            if accum_count >= grad_accum or is_last_batch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(backend.params, max_grad_norm, error_if_nonfinite=True).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                summary = {
                    "step": global_step,
                    "epoch": epoch,
                    "pairs_seen": epoch * len(pairs) + min(start_idx + bsz, len(pairs)),
                    "loss": round(sum(s["loss"] for s in accum_stats) / len(accum_stats), 6),
                    "reward_margin": round(sum(s["reward_margin"] for s in accum_stats) / len(accum_stats), 6),
                    "implicit_acc": round(sum(s["implicit_acc"] for s in accum_stats) / len(accum_stats), 4),
                    "grad_norm": round(grad_norm, 6),
                    "elapsed_s": round(time.time() - t_start, 2),
                }
                history.append(summary)
                print(json.dumps(summary), flush=True)
                accum_count = 0
                accum_stats = []

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    ckpt_path = output / f"step_{global_step:05d}.safetensors"
                    backend.save(ckpt_path, {"dpo_step": str(global_step), "source_lora": config["lora"]})

    (output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    backend.save(output / "final.safetensors", {"source_lora": config["lora"], "method": "anima-flow-dpo", "pairs": str(len(pairs))})
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    print("DPO_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
