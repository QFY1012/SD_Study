"""GRPO math: float32 stochastic flow transitions and clipped policy loss.

Matches the SDE correction and dimension-mean log density used by DanceGRPO's
FLUX implementation. This is a dimension-normalized surrogate, not the sum
log likelihood of the full latent. The terminal deterministic step is excluded.
"""
import math
import torch


def initial_noise(shape, seed, iteration, sample, same_noise=True, device='cpu'):
    noise_seed = int(seed) + int(iteration) if same_noise else int(seed) + int(iteration) * 1000003 + int(sample)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    return torch.randn(shape, device=device, generator=generator)


def transition(x, velocity, sigma, sigma_next, eta, next_x=None, generator=None):
    sigma, sigma_next = float(sigma), float(sigma_next)
    if not (0 < sigma <= 1 and 0 <= sigma_next < sigma and eta > 0):
        raise ValueError('Require 0 <= sigma_next < sigma <= 1 and eta > 0')
    x, velocity = x.float(), velocity.float()
    delta = sigma_next - sigma
    x0 = x - sigma * velocity
    score = -(x - x0 * (1 - sigma)) / sigma**2
    mean = x + delta * velocity - 0.5 * eta**2 * score * delta
    std = eta * math.sqrt(-delta)
    if next_x is None:
        next_x = mean + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * std
    target = next_x.detach().float()
    logp = (-0.5 * ((target - mean) / std)**2 - math.log(std) - 0.5 * math.log(2 * math.pi)).flatten(1).mean(1)
    return target, logp


def advantages(rewards):
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError('Need at least two finite rewards per prompt')
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def policy_loss(new_logp, old_logp, advantage, clip=1e-4, advantage_clip=5.0):
    if not 0 < clip < 1:
        raise ValueError('clip must lie in (0, 1)')
    ratio = (new_logp.float() - old_logp.detach().float()).exp()
    a = torch.clamp(advantage.detach(), -advantage_clip, advantage_clip)
    loss = torch.maximum(-a * ratio, -a * ratio.clamp(1 - clip, 1 + clip)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite GRPO loss')
    return loss, ratio.detach()
