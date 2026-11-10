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
    iterations = config.get('iterations', 1)
    eta, cfg = config.get('eta', 0.3), config.get('cfg', 4.0)
    fraction, clip = config.get('timestep_fraction', 0.25), config.get('clip_range', 1e-4)
    if width <= 0 or height <= 0 or width % 32 or height % 32 or group < 2 or steps < 2 or iterations < 1:
        raise ValueError('Invalid dimensions, group size, steps or iterations')
    if not 0 < fraction <= 1 or not 0 < eta <= 1 or not 0 < clip < 1 or not math.isfinite(cfg):
        raise ValueError('Invalid timestep_fraction, eta, clip_range or cfg')
    if not 0 < config.get('learning_rate', 1e-6) or not 0 < config.get('max_grad_norm', 1.0):
        raise ValueError('learning_rate and max_grad_norm must be positive')
    if args.validate_config:
        print('Config structure valid; weights, CUDA and model compatibility have NOT been tested')
        return
    sys.path.insert(0, str(Path(config['sd_scripts']).resolve()))
    import torch
    from .backend import AnimaBackend
    from .core import transition, advantages, policy_loss
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
    backend.cache_prompts(config['prompts'], negative)
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
        prompt = config['prompts'][iteration % len(config['prompts'])]
        trajectories, jobs = [], []
        # Keep complete group on CPU, only one sample/transition on CUDA at once.
        selected = sorted(rng.sample(range(steps - 1), train_count))
        for sample in range(group):
            sample_seed = seed + iteration * group + sample
            generator = torch.Generator(device='cuda').manual_seed(sample_seed)
            x = torch.randn((1, 16, 1, height // 8, width // 8), device='cuda', generator=generator)
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
            path = directory / f'sample_{sample:03d}.png'
            backend.save_image(x, path)
            jobs.append({'id': str(sample), 'prompt': prompt, 'image': str(path), 'seed': sample_seed})
            trajectories.append(trajectory)
        rewards = torch.tensor(score(config['reward_command'], jobs, directory), dtype=torch.float32)
        adv = advantages(rewards)
        # Before every first update, verify old/new probability agreement on the same trajectory.
        max_delta = 0.0
        with torch.no_grad():
            for trajectory in trajectories:
                t, x, target, old_logp = trajectory[0]
                v = backend.velocity(x, sigmas[t], prompt, negative, cfg)
                _, lp = transition(x.to('cuda'), v, sigmas[t], sigmas[t + 1], eta, next_x=target.to('cuda'))
                max_delta = max(max_delta, (lp.cpu() - old_logp).abs().max().item())
        if max_delta > config.get('logprob_tolerance', 0.005):
            raise RuntimeError(f'Old/new log probabilities disagree: {max_delta}')
        optimizer.zero_grad(set_to_none=True)
        before_update = [p.detach().cpu().clone() for p in backend.params] if config.get('verify_checkpoint', False) else None
        losses = []
        if adv.abs().max() > 0:
            for i, trajectory in enumerate(trajectories):
                for t, x, target, old_logp in trajectory:
                    v = backend.velocity(x, sigmas[t], prompt, negative, cfg)
                    _, lp = transition(x.to('cuda'), v, sigmas[t], sigmas[t + 1], eta, next_x=target.to('cuda'))
                    loss, _ = policy_loss(lp, old_logp.to('cuda'), adv[i].to('cuda'), clip)
                    (loss / (group * train_count)).backward()
                    losses.append(loss.item())
            if not any(p.grad is not None and p.grad.abs().max() > 0 for p in backend.params):
                raise RuntimeError('No nonzero LoRA gradients')
            if any(p.grad is not None for p in backend.model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen base model received gradients')
            torch.nn.utils.clip_grad_norm_(backend.params, config.get('max_grad_norm', 1.0), error_if_nonfinite=True)
            optimizer.step()
        backend.save(directory / 'lora.safetensors', {'grpo_iteration': str(iteration), 'source_lora': config['lora']})
        verification = {}
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
            del saved, before_update
        metrics = {'iteration': iteration, 'rewards': rewards.tolist(), 'advantages': adv.tolist(),
                   'mean_loss': sum(losses) / max(1, len(losses)), 'logprob_delta': max_delta,
                   'skipped_equal_rewards': bool(adv.abs().max() == 0), 'verification': verification,
                   'peak_cuda_memory_gb': torch.cuda.max_memory_allocated() / 1024**3}
        (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    backend.save(output / 'final.safetensors', {'source_lora': config['lora'], 'method': 'anima-grpo-experimental'})
    torch.save(optimizer.state_dict(), output / 'optimizer.pt')


if __name__ == '__main__':
    main()
