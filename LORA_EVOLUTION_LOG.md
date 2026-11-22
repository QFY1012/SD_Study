# Cheese v26 LoRA Evolution & Dataset Log

This document records the lineage of LoRA checkpoints, DPO datasets, training configurations, and evaluation runs under `D:\untig\SD_Study`.

## Dataset & Archive Convention

Each training dataset is saved under `D:\untig\SD_Study\dpo_gen_<stage>\` with:
- Raw / edited images (`*.png`)
- `prompts_used.json`
- `dpo_pairs_*_local.json` (local paths)
- `dpo_pairs_*_drive.json` (Colab/Drive paths)

---

## Evolution Tree

```text
waiANIMA_v10Base10 + cheese_v26 (最初版本)
  │
  └── [Run 3] cheese_v26_dpo_run3/step_00420.safetensors
        │
        ├── [Stage 1] ep5_hand147_frozen27 (hand147, ep=5, lr=1e-5, freeze=[27], seed=2027)
  │     │  Output: /content/drive/MyDrive/lora_training/output/cheese_v26_dpo_ep5_hand147_frozen27/final.safetensors
  │     │  Eval:   dpo_eval_ep5_hand147_frozen27_final_hand50_distill (400 imgs)
  │     │
  │     ├── [Stage 2a] facefix72 (9 fix × 8 rollouts = 72 pairs, ep=4, 72 steps, lr=5e-6)
  │     │     Output: .../cheese_v26_dpo_ep5_hand147_frozen27_facefix72/final.safetensors
  │     │     Note:   Colors faded due to 8x rollout mismatch & overfitting 9 wins.
  │     │
  │     ├── [Stage 2b] facefix1to1_reg36 (9 exact 1-to-1 × 4 + 36 hand147 reg, ep=3, 54 steps, lr=5e-6)
  │     │     Output: .../cheese_v26_dpo_ep5_hand147_frozen27_facefix1to1_reg36/final.safetensors
  │     │
  │     ├── [Stage 2c] facefix9_exact (9 exact 1-to-1, no reg, grad_accum=3, lr=5e-6)
  │     │     ├── ep=8 (24 steps): .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_exact/final.safetensors
  │     │     │     Eval: dpo_eval_ep5_hand147_frozen27_facefix9_exact_final_hand50_distill (400 imgs)
  │     │     └── +cont3 (+9 steps = 11 ep): .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_exact_cont3_1794610773/final.safetensors
  │     │
  │     └── [Stage 2d] facefix9_exact_v2 (redrawn 9 exact 1-to-1, no reg, grad_accum=3, lr=5e-6)
  │           ├── ep=6 (18 steps): .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_exact_v2_ep6/final.safetensors
  │           └── ep=12 from 0 (36 steps): .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_exact_v2_ep12_from0/final.safetensors
  │                 Eval: dpo_eval_ep5_hand147_frozen27_facefix9_exact_v2_ep12_from0_final_hand50_distill (400 imgs)
  │                 Local: D:\untig\SD_Study\dpo_gen_ep5_facefix9_v2_ep12_hand50 (picker: dpo_picker_hand_v2_ep12.html)
  │
  │                 └── [Stage 3] v2_ep12_hand100_lr5e6 (100 pairs, ep=5, 125 steps, lr=5e-6, freeze=[27], seed=2027)
  │                       Output: .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_v2_ep12_hand100_lr5e6/final.safetensors
  │                       Eval:   dpo_eval_v2_ep12_hand100_lr5e6_final_hand50_distill (400 imgs, base_seed=70000)
  │                       Local:  D:\untig\SD_Study\dpo_gen_v2_ep12_hand100_lr5e6_hand50 (picker: dpo_picker_v2_ep12_hand100_lr5e6_hand50.html)
  │
  │                       └── [Stage 4] hand49_lr5e6 (49 pairs, ep=5, ~61 steps, lr=5e-6, freeze=[27], seed=2027)
  │                             Output: .../cheese_v26_dpo_ep5_hand147_frozen27_facefix9_v2_ep12_hand100_lr5e6_hand49_lr5e6_1794618872/final.safetensors
  │                             Eval:   dpo_eval_hand49_lr5e6_final_hand50_seed80k_distill (400 imgs, base_seed=80000, running)
```

---

## Saved Local Datasets

| Stage | Local Directory | Contents | Notes |
|---|---|---|---|
| Foot 4-toes | `D:\untig\SD_Study\dpo_gen_ep5_foot_4toes_step5_50groups` | 50 groups × 8 rollouts | Foot anatomy rollouts |
| Hand & Fur (`hand147`) | `D:\untig\SD_Study\dpo_gen_ep5_hand_fur_step5_50groups` | 50 groups × 8 rollouts | Source for `ep5_hand147_frozen27` (147 pairs) |
| Face Fix 9 (`face_fix9`) | `D:\untig\SD_Study\dpo_gen_ep5_face_fix9` | 9 `*_fix.png` + 9 orig `*_rX.png`, `prompts_used.json`, `dpo_pairs_face_fix9_exact_{local,drive}.json` | Manual face-marking removal pairs |
| Face Fix v2 Ep12 (`hand100`) | `D:\untig\SD_Study\dpo_gen_ep5_facefix9_v2_ep12_hand50` | 50 groups × 8 rollouts, `dpo_picker_hand_v2_ep12.html` | Source for Stage 3 (100 pairs) |
| Hand100 lr5e-6 (`hand49`) | `D:\untig\SD_Study\dpo_gen_v2_ep12_hand100_lr5e6_hand50` | 50 groups × 8 rollouts, `dpo_picker_v2_ep12_hand100_lr5e6_hand50.html`, `dpo_pairs_local.json` | Source for Stage 4 (49 pairs) |

---

## Shared Training & Eval Defaults

- **Base Model**: `waiANIMA_v10Base10.safetensors` (`waiANIMA_v10Base10_txt.safetensors`, `qwenImage_qwenImageVAE.safetensors`)
- **DPO Core**: `beta_dpo=2000.0`, `max_grad_norm=5.0`, `freeze_blocks=[27]`, `seed=2027`, `batch_size=1`
- **Hand 50 Distill Eval**: 50 prompts × 8 rollouts (`400` images), `width=768`, `height=1024`, `steps=20`, `teacher_start_step=5`, `cfg=4.0`, `eta=0.3`, `flow_shift=5.0`, `base_seed=70000`, `same_noise=True`
