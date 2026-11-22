"""Single-GPU experimental trainer. Invoke: python -m anima_grpo.train --config ..."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import random


def score(command, jobs, directory):
    request, response = directory / 'reward_request.json', directory / 'reward_response.json'
    request.write_text(json.dumps({'samples': jobs}, ensure_ascii=False, indent=2))
    response.unlink(missing_ok=True)
    subprocess.run([*command, str(request), str(response)], check=True)
    values = json.loads(response.read_text())['scores']
    if len(values) != len(jobs) or {v['id'] for v in values} != {j['id'] for j in jobs}:
        raise ValueError('Reward response must contain every sample id exactly once')
    mapping = {v['id']: v['reward'] for v in values}
    rewards = [mapping[j['id']] for j in jobs]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rewards):
        raise ValueError('Rewards must be finite numbers')
    return rewards


def build_eval_jobs(eval_prompts: list[str], image_paths: list[Path], sample_count: int, seed: int) -> list[dict]:
    """Select a fixed eval subset and group by 6 so composite reports stay 6 columns wide."""
    k = min(max(0, sample_count), len(eval_prompts), len(image_paths))
    chosen = sorted(random.Random(seed).sample(range(len(eval_prompts)), k)) if k > 0 else []
    jobs = []
    for pos, idx in enumerate(chosen):
        jobs.append({
            'id': f'eval_{idx:03d}',
            'group_id': f'ev{pos // 6:02d}',
            'prompt': eval_prompts[idx],
            'image': str(image_paths[idx]),
            'seed': seed + idx,
        })
    return jobs


def run_eval_checkpoint(
    backend,
    eval_prompts: list[str],
    eval_sample_count: int,
    eval_command: list[str],
    negative: str,
    sigmas,
    width: int,
    height: int,
    cfg: float,
    seed: int,
    eval_dir: Path,
) -> dict:
    """Generate full eval split deterministically, then score a sampled subset with Astra."""
    import torch

    eval_dir.mkdir(parents=True, exist_ok=True)
    steps = len(sigmas) - 1
    image_paths: list[Path] = []
    for idx, prompt in enumerate(eval_prompts):
        gen = torch.Generator(device='cuda').manual_seed(seed + idx)
        x = torch.randn((1, 16, 1, height // 8, width // 8), generator=gen, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            for t in range(steps):
                v = backend.velocity(x, sigmas[t], prompt, negative, cfg)
                x = x + (sigmas[t + 1].to(x.device) - sigmas[t].to(x.device)) * v
        img_path = eval_dir / f'eval_{idx:03d}.png'
        backend.save_image(x, img_path)
        image_paths.append(img_path)

    jobs = build_eval_jobs(eval_prompts, image_paths, eval_sample_count, seed)
    rewards = score(eval_command, jobs, eval_dir) if jobs else []
    pass_count = sum(1 for r in rewards if r > 0.0)
    veto_counts = {}
    trace_file = eval_dir / 'mimo_trace.json'
    if trace_file.is_file():
        trace_data = json.loads(trace_file.read_text(encoding='utf-8'))
        for grp in trace_data.values():
            for a in grp.get('audits', {}).values():
                for key in ('severed_or_extra_limbs', 'digit_error', 'paw_pad_error', 'identity_or_clothes_error'):
                    if a.get(key):
                        veto_counts[key] = veto_counts.get(key, 0) + 1
                if not a.get('prompt_ok', True):
                    veto_counts['prompt_mismatch'] = veto_counts.get('prompt_mismatch', 0) + 1
    eval_metrics = {
        'total_eval_generated': len(image_paths),
        'astra_scored_count': len(rewards),
        'astra_pass_count': pass_count,
        'astra_pass_rate': round(pass_count / len(rewards), 4) if rewards else 0.0,
        'astra_mean_reward': round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        'veto_counts': veto_counts,
        'sampled_ids': [j['id'] for j in jobs],
    }
    (eval_dir / 'eval_metrics.json').write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
    print(json.dumps({'eval_checkpoint': eval_dir.name, **eval_metrics}), flush=True)
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-config', action='store_true', help='Validate without importing model dependencies')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    required = ['sd_scripts', 'dit', 'text_encoder', 'vae', 'lora', 'prompts', 'reward_command', 'output']
    for key in required:
        if key not in config:
            raise ValueError('Missing config key: ' + key)
    if not config['prompts'] or not all(isinstance(p, str) and p.strip() for p in config['prompts']):
        raise ValueError('prompts must contain nonempty strings')
    if not isinstance(config['reward_command'], list) or not config['reward_command'] or not all(isinstance(x, str) for x in config['reward_command']):
        raise ValueError('reward_command must be an argv list; shell commands are not accepted')
    width, height = config.get('width', 512), config.get('height', 512)
    group, steps = config.get('group_size', 4), config.get('sampling_steps', 20)
    prompts_per_step = config.get('prompts_per_step', 1)
    save_every = config.get('save_every', 1)
    eval_every = config.get('eval_every', 0)
    eval_sample_count = config.get('eval_sample_count', 24)
    eval_prompts = config.get('eval_prompts', [])
    iterations = config.get('iterations', 1)
    eta, cfg = config.get('eta', 0.3), config.get('cfg', 4.0)
    fraction, clip = config.get('timestep_fraction', 0.25), config.get('clip_range', 1e-4)
    if width <= 0 or height <= 0 or width % 32 or height % 32 or group < 2 or steps < 2 or iterations < 1:
        raise ValueError('Invalid dimensions, group size, steps or iterations')
    if isinstance(prompts_per_step, bool) or not isinstance(prompts_per_step, int) or prompts_per_step < 1:
        raise ValueError('prompts_per_step must be a positive integer')
    if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
        raise ValueError('save_every must be a positive integer')
    if isinstance(eval_every, bool) or not isinstance(eval_every, int) or eval_every < 0:
        raise ValueError('eval_every must be a non-negative integer')
    if isinstance(eval_sample_count, bool) or not isinstance(eval_sample_count, int) or eval_sample_count < 0:
        raise ValueError('eval_sample_count must be a non-negative integer')
    if eval_every > 0 and (not eval_prompts or not all(isinstance(p, str) and p.strip() for p in eval_prompts)):
        raise ValueError('eval_prompts must contain nonempty strings when eval_every > 0')
    eval_reward_command = config.get('eval_reward_command')
    if eval_reward_command is not None:
        if not isinstance(eval_reward_command, list) or not eval_reward_command or not all(isinstance(x, str) for x in eval_reward_command):
            raise ValueError('eval_reward_command must be a nonempty argv list')
    else:
        eval_reward_command = [*config['reward_command'], '--skip-pick']
    if not 0 < fraction <= 1 or not 0 < eta <= 1 or not 0 < clip < 1 or not math.isfinite(cfg):
        raise ValueError('Invalid timestep_fraction, eta, clip_range or cfg')
    if not 0 < config.get('learning_rate', 1e-6) or not 0 < config.get('max_grad_norm', 1.0):
        raise ValueError('learning_rate and max_grad_norm must be positive')
    if not isinstance(config.get('init_same_noise', True), bool):
        raise ValueError('init_same_noise must be a boolean')
    if args.validate_config:
        print('Config structure valid; weights, CUDA and model compatibility have NOT been tested')
        return
    sys.path.insert(0, str(Path(config['sd_scripts']).resolve()))
    import torch
    from .backend import AnimaBackend
    from .core import initial_noise, transition, advantages, policy_loss
    if not torch.cuda.is_available():
        raise RuntimeError('Actual Anima training requires CUDA')
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)  # refuse accidental checkpoint overwrite
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    backend = AnimaBackend(config, torch.device('cuda'))
    negative = config.get('negative_prompt', '')
    all_unique_prompts = list(dict.fromkeys([*config['prompts'], *eval_prompts]))
    backend.cache_prompts(all_unique_prompts, negative)
    optimizer = torch.optim.AdamW(backend.params, lr=config.get('learning_rate', 1e-6), weight_decay=0)
    sigmas = torch.linspace(1, 0, steps + 1)
    shift = config.get('flow_shift', 5.0)
    if shift <= 0:
        raise ValueError('flow_shift must be positive')
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    train_count = max(1, int((steps - 1) * fraction))
    for iteration in range(iterations):
        directory = output / f'iteration_{iteration:05d}'
        directory.mkdir()
        trajectories, sample_prompts, jobs = [], [], []
        # Keep complete step batch on CPU, only one sample/transition on CUDA at once.
        selected = sorted(rng.sample(range(steps - 1), train_count))
        for slot in range(prompts_per_step):
            prompt_idx = iteration * prompts_per_step + slot
            prompt = config['prompts'][prompt_idx % len(config['prompts'])]
            for sample in range(group):
                sample_seed = seed + prompt_idx * group + sample
                generator = torch.Generator(device='cuda').manual_seed(sample_seed)
                x = initial_noise((1, 16, 1, height // 8, width // 8), seed, prompt_idx, sample, same_noise=config.get('init_same_noise', True), device='cuda')
                trajectory = []
                with torch.no_grad():
                    for t in range(steps):
                        v = backend.velocity(x, sigmas[t], prompt, negative, cfg)
                        if t == steps - 1:
                            # Zero-noise endpoint: deterministic Euler step, never scored as a transition.
                            x = x + (sigmas[t + 1].to(x.device) - sigmas[t].to(x.device)) * v
                        else:
                            old_x = x
                            x, logp = transition(old_x, v, sigmas[t], sigmas[t + 1], eta, generator=generator)
                            if t in selected:
                                trajectory.append((t, old_x.cpu(), x.cpu(), logp.cpu()))
                name = f'sample_{sample:03d}.png' if prompts_per_step == 1 else f'sample_p{slot:02d}_{sample:03d}.png'
                path = directory / name
                backend.save_image(x, path)
                sid = str(sample) if prompts_per_step == 1 else f'{slot}:{sample}'
                jobs.append({'id': sid, 'group_id': f'p{slot}', 'prompt': prompt, 'image': str(path), 'seed': sample_seed})
                trajectories.append(trajectory)
                sample_prompts.append(prompt)
        rewards = torch.tensor(score(config['reward_command'], jobs, directory), dtype=torch.float32)
        adv = torch.cat([advantages(rewards[slot * group : (slot + 1) * group]) for slot in range(prompts_per_step)])
        # Before every first update, verify old/new probability agreement on the same trajectory.
        max_delta = 0.0
        with torch.no_grad():
            for sample_idx, trajectory in enumerate(trajectories):
                t, x, target, old_logp = trajectory[0]
                v = backend.velocity(x, sigmas[t], sample_prompts[sample_idx], negative, cfg)
                _, lp = transition(x.to('cuda'), v, sigmas[t], sigmas[t + 1], eta, next_x=target.to('cuda'))
                max_delta = max(max_delta, (lp.cpu() - old_logp).abs().max().item())
        if max_delta > config.get('logprob_tolerance', 0.005):
            raise RuntimeError(f'Old/new log probabilities disagree: {max_delta}')
        optimizer.zero_grad(set_to_none=True)
        before_update = [p.detach().cpu().clone() for p in backend.params] if config.get('verify_checkpoint', False) else None
        losses = []
        dead_rows = int((adv == 0).sum().item())
        total_samples = prompts_per_step * group
        if adv.abs().max() > 0:
            for i, trajectory in enumerate(trajectories):
                if adv[i].item() == 0.0:
                    continue
                for t, x, target, old_logp in trajectory:
                    v = backend.velocity(x, sigmas[t], sample_prompts[i], negative, cfg)
                    _, lp = transition(x.to('cuda'), v, sigmas[t], sigmas[t + 1], eta, next_x=target.to('cuda'))
                    loss, _ = policy_loss(lp, old_logp.to('cuda'), adv[i].to('cuda'), clip)
                    (loss / (total_samples * train_count)).backward()
                    losses.append(loss.item())
            if not any(p.grad is not None and p.grad.abs().max() > 0 for p in backend.params):
                raise RuntimeError('No nonzero LoRA gradients')
            if any(p.grad is not None for p in backend.model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen base model received gradients')
            torch.nn.utils.clip_grad_norm_(backend.params, config.get('max_grad_norm', 1.0), error_if_nonfinite=True)
            optimizer.step()
        should_save = (iteration + 1) % save_every == 0 or iteration == iterations - 1
        verification = {}
        if should_save:
            backend.save(directory / 'lora.safetensors', {'grpo_iteration': str(iteration), 'source_lora': config['lora']})
            if before_update is not None:
                verification['max_parameter_change'] = max((p.detach().cpu() - b).abs().max().item() for p, b in zip(backend.params, before_update))
                from safetensors.torch import load_file
                saved = load_file(str(directory / 'lora.safetensors'))
                current = backend.network.state_dict()
                if set(saved) != set(current):
                    raise RuntimeError('Saved/native checkpoint keys differ')
                for name, value in current.items():
                    torch.testing.assert_close(saved[name], value.detach().cpu(), rtol=0, atol=0)
                backend.network.load_state_dict(saved, strict=True)
                verification['native_reload'] = 'passed'
                del saved
        if before_update is not None:
            del before_update
        metrics = {'iteration': iteration, 'prompts_per_step': prompts_per_step, 'saved_checkpoint': should_save, 'rewards': rewards.tolist(), 'advantages': adv.tolist(),
                   'dead_rows_ratio': dead_rows / total_samples, 'mean_loss': sum(losses) / max(1, len(losses)), 'logprob_delta': max_delta,
                   'skipped_equal_rewards': bool(adv.abs().max() == 0), 'verification': verification,
                   'peak_cuda_memory_gb': torch.cuda.max_memory_allocated() / 1024**3}
        (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
        if eval_every > 0 and (iteration + 1) % eval_every == 0:
            eval_dir = output / f'eval_{iteration + 1:05d}'
            run_eval_checkpoint(
                backend,
                eval_prompts,
                eval_sample_count,
                eval_reward_command,
                negative,
                sigmas,
                width,
                height,
                cfg,
                seed + 10000,
                eval_dir,
            )
    backend.save(output / 'final.safetensors', {'source_lora': config['lora'], 'method': 'anima-grpo-experimental'})
    torch.save(optimizer.state_dict(), output / 'optimizer.pt')


if __name__ == '__main__':
    main()
