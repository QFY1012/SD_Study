# 用 Colab 生图(Anima + LoRA)

前提:已用 `colab` 命令行登录、开好 GPU 会话并挂载 Drive(见 colab skill)。下面的 `work` 换成你的会话名。
模型默认读 Drive 的 `MyDrive/ComfyUI/models/`:
`diffusion_models/waiANIMA_v10Base10.safetensors`、`text_encoders/waiANIMA_v10Base10_txt.safetensors`、
`vae/qwenImage_qwenImageVAE.safetensors`、`loras/cheesev1.safetensors`。别的模型/LoRA:改 `vm_generate.py` 顶部的 `DEFAULTS`,或在任务文件里逐项覆盖。

```bash
colab exec -s work -f vm_install.py     # 克隆 ComfyUI 并后台安装依赖
colab exec -s work -f vm_status.py      # 重复运行,直到出现 "install: done"
colab exec -s work -f vm_start.py       # 启动 ComfyUI(第一次要 30 秒左右)
colab upload -s work gen_jobs.example.json /content/gen_jobs.json   # 改好提示词再传
colab exec -s work -f vm_generate.py    # 提交生图任务
colab exec -s work -f vm_status.py      # 看进度/结果(第一次要几分钟加载模型,之后每张约 70 秒)
colab ls -s work /content/test_out/cheese_sports
colab download -s work /content/test_out/cheese_sports/t01_soccer_00001_.png ./t01.png
```

任务文件格式:`{"defaults": {...}, "jobs": [{"name": "...", "positive": "...", "seed": 101}]}`。
可覆盖的项:`negative`、`seed`、`width`/`height`(默认 768x1344)、`steps`(30)、`cfg`(4.5)、`sampler`(er_sde)、`scheduler`(simple)、`unet`、`clip`、`vae`、`lora`、`lora_strength`、`prefix`(输出子目录)。
`lora` 设为空字符串则不加 LoRA。

提示词要点:标签全小写、空格代替下划线;质量前缀 `masterpiece, best quality, score_9, score_8, score_7`;画师名前加 `@`;
cheesev1 学的基线是裸体,生成穿衣服的图要在负面里加 `nude, nsfw`。

注意:`colab exec` 一分钟左右会超时但代码仍在跑;生图任务排在 ComfyUI 队列里,慢是正常的,用 `vm_status.py` 查看。
