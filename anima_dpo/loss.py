"""Rectified-flow Diffusion-DPO math primitives aligned with Anima (kohya-ss/sd-scripts)."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def sample_flow_sigmas(
    batch_size: int,
    *,
    scheme: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    flow_shift: float = 5.0,
    min_sigma: float = 0.02,
    max_sigma: float = 0.98,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample flow sigmas in (min_sigma, max_sigma) with optional logit-normal + SD3 shift."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    if not (0.0 < min_sigma < max_sigma < 1.0):
        raise ValueError("Require 0 < min_sigma < max_sigma < 1")

    if scheme == "uniform":
        u = torch.rand((batch_size,), generator=generator, device=device, dtype=torch.float32)
    elif scheme == "logit_normal":
        z = torch.randn((batch_size,), generator=generator, device=device, dtype=torch.float32)
        u = torch.sigmoid(logit_mean + logit_std * z)
    else:
        raise ValueError(f"Unsupported sigma sampling scheme: {scheme!r}")

    sigmas = (u * flow_shift) / (1.0 + (flow_shift - 1.0) * u)
    return sigmas.clamp(min=min_sigma, max=max_sigma)


def flow_noisy_latent(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Rectified Flow noisy state x_t = (1 - sigma)*x0 + sigma*noise and velocity target = noise - x0."""
    if x0.shape != noise.shape:
        raise ValueError(f"Shape mismatch: x0 {tuple(x0.shape)} vs noise {tuple(noise.shape)}")
    s = sigma.float().view(x0.shape[0], *([1] * (x0.ndim - 1)))
    x0_f = x0.float()
    noise_f = noise.float()
    xt = (1.0 - s) * x0_f + s * noise_f
    target = noise_f - x0_f
    return xt, target


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return float32 mean squared error per batch element."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)


def dpo_loss(
    mse_policy_w: torch.Tensor,
    mse_policy_l: torch.Tensor,
    mse_ref_w: torch.Tensor,
    mse_ref_l: torch.Tensor,
    *,
    beta: float = 2000.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Diffusion-DPO loss (-log sigmoid(-0.5 * beta * ((mse_w - ref_w) - (mse_l - ref_l))))."""
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")

    diff_w = mse_policy_w.float() - mse_ref_w.detach().float()
    diff_l = mse_policy_l.float() - mse_ref_l.detach().float()
    # Lower MSE relative to ref means higher implicit reward.
    model_diff = diff_w - diff_l
    logits = -0.5 * beta * model_diff
    loss = (-F.logsigmoid(logits)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DPO loss")

    with torch.no_grad():
        rew_w = (-0.5 * beta * diff_w).mean().item()
        rew_l = (-0.5 * beta * diff_l).mean().item()
        acc = (model_diff < 0).float().mean().item()
    metrics = {
        "loss": float(loss.item()),
        "implicit_reward_w": rew_w,
        "implicit_reward_l": rew_l,
        "reward_margin": rew_w - rew_l,
        "implicit_acc": acc,
    }
    return loss, metrics
