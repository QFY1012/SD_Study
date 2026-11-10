# SD_Study

以 Stable Diffusion 为代表的文生图模型发展调研。

- 📄 调研报告：[stable_diffusion_image_generation_survey.md](stable_diffusion_image_generation_survey.md)
- 🌐 可视化讲解页（本地用浏览器打开）：[sd_explainer.html](sd_explainer.html)
- 🧪 训练目标交互讲解页：[training_objectives.html](training_objectives.html)
- 🧩 Anima 架构可视化：[anima_architecture.html](anima_architecture.html)
- 📊 开源/开放权重模型发布时间线数据：[data/open_image_models_timeline.csv](data/open_image_models_timeline.csv)

调研截止：2026-11-02。2026 年条目以 Hugging Face 官方仓库元数据为准，商用前请核对各模型 LICENSE。

## Anima 训练与偏好优化

- [SFT、GRPO 与基线出图使用说明](ANIMA_RL_SFT_GUIDE.md)
- [SFT Colab notebook](notebooks/ANIMA_Trainer_v7.ipynb)
- [Anima GRPO 实验实现](anima_grpo/README.md)
- [600 条任务集](data/anima_tasks_600/README.md) · [交互式 t-SNE](data/anima_tasks_600/tsne.html)
