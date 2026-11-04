# ComfyUI-SEGSInpaint

独立实现的两个节点,用来在远端 ComfyUI 里直接跑「框选 → 局部重绘」流程(不依赖 Impact Pack、SAM 或远程插件):

- `SAM3BoxesToSEGS`:`CreateBoundingBoxes` 画的框 → SEGS。框本身就是蒙版,`padding` 向外扩蒙版,`crop_factor` 把扩后的框按中心放大得到上下文裁剪区。
- `RemoteSEGSDetailer`:对每个 SEGS 裁剪 → 按 `guide_size`/`max_size` 缩放(边长取 8 的倍数)→ 带蒙版 img2img → 缩回并按 `feather` 羽化贴回。`url`/`token` 仅为兼容保留,被忽略。

类名和输入名与原工作流一致,PNG 里的 API 工作流可直接加载。

已用假 VAE/采样器验证:框解析与裁剪、缩放尺寸、羽化贴回、无框 no-op、`force_inpaint=False` 跳过。**尚未在真实 SDXL 模型上跑过。**
