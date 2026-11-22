# Experimental Anima GRPO adapter

This is a single-CUDA-device prototype, validated with a one-update GPU smoke test; it is not a quality-training recipe.
It reuses native sd-scripts Anima model/LoRA loading, supports an existing character
LoRA, freezes the base model, Qwen3 and LLM Adapter, and updates DiT LoRA only.
The complete native LoRA (including frozen adapter/text-encoder weights) is saved.

## Setup

Install the requirements of sd-scripts in a CUDA environment. Code was inspected
against sd-scripts commit `cb70b96` and DanceGRPO `15cc71d`. Do not assume arbitrary
upstream versions have compatible interfaces. No dependency installation or GPU
allocation is performed by this entry point.

From the SD_Study repository root:

```bash
python -m anima_grpo.train --config examples/anima/grpo.json --validate-config
python -m anima_grpo.train --config examples/anima/grpo.json
```

Edit model paths, `sd_scripts`, `output`, prompts and `reward_command` first.
The output directory must not already exist. The example is a 512px, one-iteration
smoke configuration. It is not a quality-training recommendation. Start at a small
size and measure actual peak VRAM before increasing resolution/group size.

## Reward contract

`reward_command` is an argv array, run without a shell. Two positional arguments
are appended: request JSON path and response JSON path. The request contains:

```json
{"samples": [{"id": "0", "prompt": "...", "image": "/absolute/path.png", "seed": 42}]}
```

The scorer must return every id once, with finite scalar rewards:

```json
{"scores": [{"id": "0", "reward": 0.75}]}
```

No VLM provider, API credentials or anatomy rubric is supplied yet. A VLM scorer
can write component judgments into another file and return the aggregate here.
`examples/anima/smoke_reward.py` is explicitly diagnostic brightness reward only.
It can exercise training plumbing but cannot measure fingers, toes or preference.

## Algorithm and memory

- Rollout uses the Anima velocity interface, shifted sigma schedule and CFG.
- Rollout shares initial latent noise across a prompt group by default (`init_same_noise: true`) and uses DanceGRPO's stochastic flow/SDE correction for nonterminal transitions.
- Store selected transitions and old mean-per-coordinate log densities on CPU.
- The final step is deterministic and excluded from policy loss.
- Score a full prompt group; normalize rewards with sample standard deviation.
- Verify unchanged-policy log density agreement before updating.
- Replay the same CFG, timestep and SDE kernel with gradients enabled.
- Average clipped GRPO losses across the group and selected timesteps; take one
  optimizer step per group. Equal-score groups are skipped.
- LoRA stays in eval mode during rollout AND replay to avoid dropout mismatch;
  eval mode does not disable gradients. No explicit KL penalty is implemented.

The log density uses a dimension mean, matching the upstream DanceGRPO FLUX code,
not a sum over the full latent. Retain that convention when choosing clip values.

## Verification and current limits

```bash
python -m unittest discover -s tests/anima -v
```

CPU tests cover replay agreement, a Normal-distribution density reference,
positive/negative advantage clipping, equal rewards, gradient flow only to a toy
adapter, invalid noise schedules, and reward response validation.

GPU smoke test passed on NVIDIA A100-SXM4-40GB: 256x256, two candidates, six sampling steps, one optimizer update, diagnostic brightness reward. Real checkpoint strict load, old/new log-density equality (max difference 0), nonzero LoRA update (max change 1.000240445e-6), native save/reload and inference through unmodified sd-scripts passed. Peak allocated CUDA memory was 6.566 GiB. Saved weights remain float32 (about 138 MB). These checks do not establish improvement in anatomy or preference. No production
resume loop, block swapping, distributed training, gradient checkpointing or
multi-provider VLM integration is implemented. The prototype saves optimizer
state at completion but does not claim resumable GRPO state management.

The provided output should be evaluated with the usual deterministic Anima
sampler as well as the stochastic training sampler before judging improvement.
