# Hand Reward Model & Auto-DPO Pipeline (`SAM3.1` + `DINOv3 Spatial2D`)

统一复用 ComfyUI `.venv` 运行（无需额外虚拟环境）：
`D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe`

## 1. 训练 / 增量更新手部头

```powershell
$py = "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"
& $py D:\untig\SD_Study\scripts\hand_rm\train_hand_rm.py `
  --pairs-files "d:\Download\dpo_pairs (7).json" "d:\Download\dpo_pairs (8).json" "d:\Download\dpo_pairs (1).json"
```

- 自动增量调用 `sam3.1_multiplex_fp16.safetensors` 检测 `"hand"` bbox 并裁剪缓存；
- 提取 DINOv3 `CLS + 24×24` 空间特征后训练 `Spatial2DRewardHead`（当前 274 对下验证集 `90.4%`，`margin>=2.0` 下 `93.1%`）；
- 权重输出到 `D:\untig\SD_Study\models\dinov3_hand_spatial2d_head.pt`。

## 2. 自动为新评测图打分并生成 DPO 数据

```powershell
$py = "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"
& $py D:\untig\SD_Study\scripts\hand_rm\auto_dpo_pairs.py `
  --img-dir "D:\untig\SD_Study\dpo_gen_hand49_lr5e6_hand50_seed80k" `
  --prompts-json "D:\untig\SD_Study\dpo_gen_ep5_facefix9_v2_ep12_hand50\prompts_used.json" `
  --drive-dir "/content/drive/MyDrive/lora_training/dpo_eval_hand49_lr5e6_final_hand50_seed80k_distill" `
  --min-margin 2.0
```

- 默认基线 `--min-margin 2.0`（只保留高区分度配对）；
- 输出 `dpo_pairs_auto_local.json` 与 `dpo_pairs_auto_drive.json`。
