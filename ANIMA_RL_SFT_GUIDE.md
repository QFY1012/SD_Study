# 使用本仓库做 Anima SFT、GRPO 与基线出图

## 当前代码做到哪一步

| 工作流 | 入口 | 验证情况 |
| --- | --- | --- |
| 图片与标注监督微调（SFT） | `notebooks/ANIMA_Trainer_v7.ipynb` | 同源 notebook 已用于角色 LoRA 训练；本次入库版清除了输出、暴露 seed 参数并固定依赖提交，需按自己的路径运行 |
| Anima LoRA GRPO 原型 | `python -m anima_grpo.train` | A100 40GB 上完成一次 256×256、2 候选、6 步的真实更新；概率重算、非零参数更新、原生权重重载和原版推理均通过 |
| 600 条任务基线生成 | `scripts/anima_baseline600.py` | 新增逐图保存和断点补跑入口，已做语法检查；整批 600 图尚未验证完成 |
| VLM 偏好奖励 | 外部 `reward_command` | 只有接口和 rubric 草案，尚未接入、校准正式图片评分模型 |

GRPO 的成功验证使用平均亮度奖励，仅用于检查代码通路。不能据此声称手脚或角色偏好已经改善。

## 1. 环境与模型

以 Colab 为例，在启动 GPU 后先挂载 Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

安装仓库与固定版本 sd-scripts。在新的 Colab 单元格中运行：

```bash
%%bash
set -e
cd /content
git clone --branch claude/practical-thompson-6pq2ot https://github.com/QFY1012/SD_Study.git
git clone https://github.com/kohya-ss/sd-scripts.git
git -C /content/sd-scripts checkout --detach cb70b96858477e716a2096292ab8aab433277f38
cd /content/sd-scripts
python -m pip install -r requirements.txt
python -m pip install toml pillow
```

如果目录已经存在，先确认目录和分支，不要重复 clone。模型和数据不包含在 Git 仓库内。示例使用：

```text
MyDrive/ComfyUI/models/diffusion_models/waiANIMA_v10Base10.safetensors
MyDrive/ComfyUI/models/text_encoders/waiANIMA_v10Base10_txt.safetensors
MyDrive/ComfyUI/models/vae/qwenImage_qwenImageVAE.safetensors
```

在配置中替换成自己的完整路径。启动 GPU 会消耗 Colab 计算单元；任务完成后断开并删除运行时。所有示例都应使用新的输出目录。

## 2. SFT：先学会角色，再处理偏好

SFT 数据由图片与同名 `.txt` 标注构成：

```text
dataset/
  image_001.png
  image_001.txt
  image_002.jpg
  image_002.txt
```

在 Colab 打开 `notebooks/ANIMA_Trainer_v7.ipynb`，依次运行环境安装、训练设置、数据预览与训练单元格。训练设置示例：

```python
project_name = "cheese_sft_run001"
image_directory = "/content/drive/MyDrive/lora_training/datasets/cheese_v24_train"
output_directory = "/content/drive/MyDrive/lora_training/output/cheese_sft_run001"
network_dim = 24
network_alpha = 24
learning_rate = 0.0001
training_seed = 42
max_train_epochs = 20
train_batch_size = 4
resolution = 768
enable_bucket = True
allow_image_upscale = True
```

分桶按宽高比分组；允许放大是普通插值缩放，不是生成式超分。训练不会修改原始数据集图片。`resolution=768` 是目标训练分辨率设置，实际桶宽高按宽高比生成，并非所有图片都裁成 768×768。

### SFT 中断续跑

保留数据集、batch、repeats 和项目名称，选择：

```python
resume_mode = "指定 state 文件夹"
resume_path = "/content/drive/MyDrive/lora_training/output/cheese_sft_run001/cheese_sft_run001-000012-state"
max_train_epochs = 20
```

这会恢复模型、优化器、调度器及训练进度。已有 12 轮时，目标 20 轮表示补完后 8 轮。状态目录需包含 `model.safetensors`、`optimizer.bin`、`scheduler.bin`、随机状态和 `train_state.json`。

### 降低学习率再学 20 轮

若已训练 20 轮，希望从最终权重再训练 20 轮，同时重设学习率：

```python
project_name = "cheese_continue_lr1e5_run001"
resume_mode = "从 LoRA 文件继续"
resume_path = "/content/drive/MyDrive/lora_training/output/cheese_sft_run001/cheese_sft_run001.safetensors"
learning_rate = 0.00001
max_train_epochs = 20
```

这一阶段从第 1 轮计数，优化器与学习率调度重新建立，累计相当于学了 40 轮。不要把“恢复旧优化器 state”与“重置学习率的新阶段”混淆。用独立项目名避免覆盖旧权重。

## 3. 先给任务集生成基线图

`data/anima_tasks_600/` 包含 600 条任务：480 条 train、120 条 eval。还包含 384 维 embedding、t-SNE、最近邻清单、生成与文本审查记录。

- 固定角色、五指、四趾、黄色肉垫与赛璐珞风格。
- 任务对的 embedding 余弦相似度均低于 0.84；不构成语义绝对不重复的证明。
- eval 是独立提示词留出，未保证动作家族完全隔离。
- `rubric.json` 是待人工校准的规则草案。

修改 `examples/anima/baseline600.json` 中的模型、LoRA、任务集与输出路径。默认 768×1024、28 步、CFG 4、LoRA 权重 1，使用确定性 flow Euler 出图，而非 GRPO 随机转移。

```bash
%%bash
set -e
cd /content/SD_Study
python -u scripts/anima_baseline600.py --config examples/anima/baseline600.json
```

脚本一次加载模型，按任务逐张生成并立即保存：

```text
output/
  run_config.json
  tasks_all.jsonl
  progress.json
  images/cheese_rl_0001.png
  metadata/cheese_rl_0001.json
  ...
```

元数据包含任务 ID、提示词、种子、LoRA 路径与运行指纹。重启运行时后用同一个配置重跑，会验证已有图片和元数据，再跳过完成的任务；若配置、源 LoRA 或任务集发生变化，会拒绝混用已有结果。完整的身份配置不同，应使用新目录。

eval 使用任务记录中的第一个固定种子；train 使用任务 ID 的稳定哈希种子。这一阶段只建立基线，不会更新 LoRA，也不评分。建议先运行并查看 `progress.json` 与前几张结果，确认路径与画质再等待整批完成。

## 4. GRPO：冻结基础模型，优化 DiT LoRA

当前原型冻结 Anima 基础模型、Qwen3 和 LLM Adapter，只更新 DiT LoRA。已有 LoRA 中的冻结权重也保存在完整输出中。采样轨迹放到 CPU，每次在 GPU 重算一个转移步骤。

先复制 `examples/anima/grpo.json` 为自己的配置，修改路径和输出目录。`output` 必须不存在。以下命令验证配置结构，不会加载模型：

```bash
cd /content/SD_Study
python -m anima_grpo.train --config examples/anima/grpo.json --validate-config
```

### 4.1 最小流程验证

配置中的 `reward_command` 默认指向 `examples/anima/smoke_reward.py`，它给图片平均亮度打分。只用于检查采样、评分、反向传播与保存，不能拿它训练正式角色偏好。

降低验证规模并启用保存重载检查：

```python
import json
from pathlib import Path
p = Path('/content/SD_Study/examples/anima/grpo.json')
c = json.loads(p.read_text())
c.update(width=256, height=256, group_size=2,
         sampling_steps=6, iterations=1, timestep_fraction=0.4,
         verify_checkpoint=True,
         output='/content/anima_grpo_smoke_run001')
Path('/content/grpo_smoke.json').write_text(json.dumps(c, indent=2))
```

```bash
%%bash
set -e
cd /content/SD_Study
python -u -m anima_grpo.train --config /content/grpo_smoke.json
```

检查每轮 `metrics.json`：

- `logprob_delta`：更新前同一轨迹的旧、新对数概率应接近 0。
- `verification.max_parameter_change`：有有效优势时，参数应产生非零变化。
- `verification.native_reload`：保存并重新加载原生 LoRA 的检查结果。
- `peak_cuda_memory_gb`：峰值已分配显存，数值实际上按 GiB 计算。

实际 A100 验证中，上述最小规模峰值为约 6.57 GiB，最大参数变化约 `1e-6`。这不代表 768×1024 或更大组数也只需要同样显存。平均初始 loss 为 0 可以与非零梯度并存，因为组内优势均值为 0，不同样本的梯度却不相同。

### 4.2 接入 VLM rubric

先让用户判断一批基线图，再核对 VLM 的指头计数、遮挡判断和肉垫颜色评分。无法看清的数量必须记录为无法判断，不能默认正确；任务明确要求可见却被遮挡时，降低可见性/任务完成度。

正式评分器由你实现为外部命令，例如：

```json
"reward_command": ["python", "/content/score_rubric.py"]
```

训练器会追加两个参数：输入 request JSON 路径和输出 response JSON 路径。请求格式：

```json
{"samples": [{"id": "0", "prompt": "...", "image": "/absolute/path/sample.png", "seed": 42}]}
```

返回格式：

```json
{"scores": [{"id": "0", "reward": 0.75}]}
```

必须对每个样本 ID 返回恰好一个有限数值；`NaN`、缺失或重复 ID 会报错。可根据完整 prompt 在任务集中查到对应 task ID、可见性要求和 scoring_targets。评分器可另存各分项与证据，交给 GRPO 的是聚合后的奖励。当前仓库没有正式 VLM 评分实现，也没有宣称 DeepSeek 文本模型能直接看图。

### 4.3 使用训练任务，保留评估集

将配置中的 `prompts` 改成训练集的完整 prompt：

```python
import json
from pathlib import Path
root = Path('/content/SD_Study')
c = json.loads(Path('/content/grpo_smoke.json').read_text())
tasks = [json.loads(x) for x in (root/'data/anima_tasks_600/train.jsonl').read_text().splitlines()]
c['prompts'] = [t['prompt'] for t in tasks]
c['negative_prompt'] = tasks[0]['negative_prompt']
c['reward_command'] = ['python', '/content/score_rubric.py']  # 需先完成并校准
c['output'] = '/content/drive/MyDrive/lora_training/output/cheese_grpo_run001'
c['iterations'] = 480
Path('/content/grpo_train.json').write_text(json.dumps(c, ensure_ascii=False, indent=2))
```

每个 iteration 处理一个 prompt，按列表顺序循环，因此 480 次表示 480 条训练提示词各一次；它不等于 SFT 的 480 个 epoch。每条 prompt 生成 `group_size` 张候选图，组大小增加会增加生成与评分开销。

正式分辨率、采样步数、学习率、组数应从小规模实测逐步调整。先运行少量 iteration，检查奖励是否可信、参数是否更新、显存是否足够，再增加规模。不要把 eval 放入 GRPO 训练或奖励调参。

原型每个 iteration 保存 LoRA 和指标，最终保存 `final.safetensors` 与 optimizer。**尚未实现完整 GRPO 断点状态恢复**；从已有 LoRA 开始新 run 可以继续优化，但不等于恢复随机状态、旧 rollout、优化器和调度进度。SFT notebook 的 state 续训功能不适用于这里。

## 5. 用原版推理检查新 LoRA

SFT 和 GRPO 输出均采用 sd-scripts 原生 Anima LoRA 格式。先用同一提示词、种子和参数对比原 LoRA 与新 LoRA：

```bash
%%bash
set -e
cd /content/sd-scripts
python anima_minimal_inference.py \
  --dit /content/drive/MyDrive/ComfyUI/models/diffusion_models/waiANIMA_v10Base10.safetensors \
  --text_encoder /content/drive/MyDrive/ComfyUI/models/text_encoders/waiANIMA_v10Base10_txt.safetensors \
  --vae /content/drive/MyDrive/ComfyUI/models/vae/qwenImage_qwenImageVAE.safetensors \
  --qwen_image_vae_2d \
  --lora_weight /content/anima_grpo_smoke_run001/final.safetensors \
  --lora_multiplier 1 \
  --prompt '@cheese, solo, standing, open palms, yellow paw pads, cel shading' \
  --negative_prompt 'worst quality, low quality, score_1, score_2, score_3' \
  --image_size 1024 768 --infer_steps 28 --guidance_scale 4 --seed 101 \
  --attn_mode torch --save_path /content/grpo_comparison
```

完整评估应使用 `eval.jsonl` 的任务和固定种子，检查角色保持、画风、手脚正确率与可见性，避免只追求评分器的总分。

## 6. 本地测试与任务集复现

安装 CPU PyTorch 后，在仓库根目录运行：

```bash
python -m unittest discover -s tests/anima -v
python -m compileall -q anima_grpo scripts
```

任务集验证与重建另需：

```bash
pip install numpy scikit-learn sentence-transformers matplotlib
python data/anima_tasks_600/validate.py
```

`build.py` 会调用 DeepSeek 生成和审查候选任务，使用环境变量 `DEEPSEEK_API_KEY`，模型为 `deepseek-flash`；保留的 raw/review 文件可复用，缺失批次才会重新调用 API 并产生费用。embedding 模型和包版本记录于 `report.json`，首次运行需要下载公开模型。

```bash
python data/anima_tasks_600/build.py
python data/anima_tasks_600/finalize.py
python data/anima_tasks_600/validate.py
```

`finalize.py` 打包之前已有的验证报告；重建后建议先 validate，再 finalize 一次，让压缩包携带最新报告。任务集的人工图像校准还未完成。密钥、模型权重、运行时输出与模型缓存不入库。
