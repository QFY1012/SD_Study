# 以 Stable Diffusion 为代表的文生图模型发展调研

> 调研截止：2026-11-02
> 范围：扩散模型的技术演进；Stable Diffusion 系列的完整发布史；2022–2026 年主要开源 / 开放权重文生图模型的发布情况（时间、机构、规模、架构、许可证）；闭源模型对照；生态、法律与趋势。
> 配套数据表：[`data/open_image_models_timeline.csv`](data/open_image_models_timeline.csv)

---

## 目录

1. [结论摘要](#1-结论摘要)
2. [前史：从 GAN 到潜在扩散（2014–2022）](#2-前史从-gan-到潜在扩散20142022)
3. [Stable Diffusion 系列完整发布史](#3-stable-diffusion-系列完整发布史)
4. [开源文生图模型发布全景（按阶段）](#4-开源文生图模型发布全景按阶段)
5. [技术演进主线](#5-技术演进主线)
6. [许可证演变](#6-许可证演变)
7. [闭源模型对照时间线](#7-闭源模型对照时间线)
8. [社区生态与工具链](#8-社区生态与工具链)
9. [数据、版权与监管](#9-数据版权与监管)
10. [格局与趋势判断](#10-格局与趋势判断)
11. [选型建议（截至 2026-11）](#11-选型建议截至-2026-11)
12. [方法说明与信息源](#12-方法说明与信息源)

---

## 1. 结论摘要

- **起点**：2022 年 8 月 22 日 Stable Diffusion v1.4 公开权重，是文生图从闭源走向开放的分水岭。它的基础是 CompVis（LMU 慕尼黑）的潜在扩散模型 LDM，在压缩后的潜空间里做扩散，消费级显卡就能跑。
- **Stability AI 的高峰与退潮**：SD 1.5（2022.10）和 SDXL（2023.07）定义了社区生态。SD 2.x 因数据过滤和改用 OpenCLIP 遭到社区冷遇；SD3 Medium（2024.06）因许可证和人体生成质量问题受挫；SD 3.5（2024.10）之后，Stability 再没有发布新的开放文生图基座。截至 2026-11，SD 3.5 仍是其最新开放图像模型，公司重心转向企业服务、音频（Stable Audio 3.0，2026.05）和行业合作。
- **主导权转移**：LDM / SD 的核心作者 Rombach、Blattmann、Esser 于 2024 年离开 Stability，创立 Black Forest Labs（BFL）。FLUX.1（2024.08）接棒成为开源社区的事实标准；之后又有 FLUX.1 Kontext（2025.06）、FLUX.2（2025.11）、FLUX.2 klein（2026.01）、FLUX.3 Image Klein（2026.10）。
- **中国机构崛起**：2025 年起，开放权重的第一梯队大量来自中国机构，包括阿里 Qwen / 通义（Qwen-Image、Z-Image）、腾讯（HunyuanImage 3.0）、百度（ERNIE-Image）、智谱（CogView4、GLM-Image）、美团（LongCat-Image）、HiDream、蚂蚁（Ming / LLaDA-Image）、商汤（SenseNova-U1）。其中多数采用 Apache-2.0 或 MIT 许可。
- **架构演进**：U-Net 换成 DiT，再到 MMDiT 和单流 DiT；训练目标从 ε 预测换成 v 预测，再到 Rectified Flow / Flow Matching；文本编码器从 CLIP 换成 T5，再到 LLM / VLM；2025–2026 年出现自回归与扩散混合、统一多模态（MoT）、无 VAE 的像素级统一 Transformer 等路线。
- **规模先涨后收**：参数量从 0.86B（SD1.5 U-Net）一路涨到 12B（FLUX.1）、20B（Qwen-Image）、32B（FLUX.2 dev）、80B MoE（HunyuanImage 3.0）。2026 年又回落到 4–9B 的高效模型（Z-Image 6B、FLUX.2 klein 4B、ERNIE-Image 8B、Qwen-Image-2.1 7B、FLUX.3 Image Klein 5.6B、Mage-Flow 4B），配合 4–8 步蒸馏。
- **许可证分化**：2026 年出现明显的收紧趋势。Qwen-Image-2.1 从前代的 Apache-2.0 改为 Qwen Research License，Ideogram 4、Krea 2、FLUX.3 Klein 都用自定义许可证。宽松许可目前主要靠百度、通义 Z-Image、HiDream、商汤、蚂蚁和 BFL 的 4B 小模型维持。
- **差距**：开放模型与最强闭源模型（GPT Image 2/2.5、Nano Banana 2.x、Midjourney V8、Seedream 5）在 Artificial Analysis 竞技场上仍有约 100–160 Elo 的差距（不同快照数据不一）。不过在中文和长文本渲染、可微调性、本地部署方面，开放模型已经有不可替代的优势。

---

## 2. 前史：从 GAN 到潜在扩散（2014–2022）

| 时间 | 里程碑 | 意义 |
|---|---|---|
| 2013.12 | VAE（Kingma & Welling） | 变分自编码器，后来成为 LDM 的"压缩器" |
| 2014.06 | GAN（Goodfellow 等） | 生成对抗网络，主导 2014–2020 年的图像生成 |
| 2015.03 | Sohl-Dickstein 等提出扩散概率模型 | 扩散模型的理论起点 |
| 2017.11 | VQ-VAE | 离散视觉 token，为自回归生图打基础 |
| 2018–2020 | BigGAN、StyleGAN 1/2 | GAN 在人脸等窄域达到照片级 |
| 2019.07 | NCSN / Score Matching（Song & Ermon） | 基于分数的生成模型 |
| 2020.06 | **DDPM**（Ho 等） | 扩散模型首次在质量上与 GAN 相当 |
| 2020.10 | DDIM | 确定性采样，大幅减少步数 |
| 2020.11 | Score SDE（Song 等） | 用 SDE / ODE 统一扩散与分数模型 |
| 2020.12 | VQGAN（Esser, Rombach, Ommer） | 后来 SD 团队的前作 |
| 2021.01 | **DALL·E**（OpenAI，自回归）、**CLIP** | 文生图进入大众视野；CLIP 成为文本条件的关键 |
| 2021.05 | ADM "Diffusion Models Beat GANs" | 扩散在 ImageNet 上超越 GAN；提出 classifier guidance |
| 2021 年中 | VQGAN+CLIP、Disco Diffusion 等社区玩法 | 开源"AI 艺术"萌芽 |
| 2021.12 | **Classifier-Free Guidance**（Ho & Salimans，NeurIPS 2021 Workshop） | 之后几乎所有文生图模型的标配 |
| 2021.12 | **GLIDE**（OpenAI） | 文本条件的像素扩散 |
| 2021.12 | **LDM《High-Resolution Image Synthesis with Latent Diffusion Models》**（Rombach、Blattmann、Lorenz、Esser、Ommer），CVPR 2022 | **Stable Diffusion 的直接论文基础**：VAE 压缩到潜空间，加 U-Net 去噪和交叉注意力文本条件 |
| 2022.03 | LAION-5B 数据集 | 58.5 亿图文对，SD 的训练数据来源 |
| 2022.04 | DALL·E 2（unCLIP） | 闭源，引爆公众关注 |
| 2022.05 | Imagen（Google） | 证明大型冻结文本编码器（T5-XXL）的重要性 |
| 2022.06 | Parti（Google，自回归） | 自回归路线的代表 |
| 2022.07 | Midjourney 开放公测；HF `diffusers` 库发布 | |

**为什么是 LDM / SD 胜出**：在 f=8 的潜空间做扩散（512² 图像对应 64²×4 的潜变量），计算量降低一个数量级，使 **约 1B 参数的模型在消费级 GPU（约 10GB 显存）上推理** 成为可能。再加上 Stability AI 出资训练、权重完全公开，以及 CreativeML OpenRAIL-M 这种相对宽松的许可证，三者叠加形成了爆发。

---

## 3. Stable Diffusion 系列完整发布史

### 3.1 版本时间线

| 版本 | 发布日期 | 发布方 | 参数规模（去噪网络） | 文本编码器 | 原生分辨率 | 许可证 | 关键点 |
|---|---|---|---|---|---|---|---|
| SD v1.1–1.3 | 2022.08（研究者预览） | CompVis / Stability / Runway | U-Net 约 860M | CLIP ViT-L/14（123M） | 512² | CreativeML OpenRAIL-M | LAION-2B-en / LAION-Aesthetics 训练 |
| **SD v1.4** | **2022-08-22** | CompVis（Stability 出资，Runway 参与） | 860M | CLIP ViT-L/14 | 512² | CreativeML OpenRAIL-M | **首个公开权重版本**；4000 张 A100 规模集群训练 |
| **SD v1.5** | 2022-10-20 | RunwayML | 860M | CLIP ViT-L/14 | 512² | CreativeML OpenRAIL-M | 社区最长寿的基座。2024.08 Runway 删除了 HF 仓库，现由 `stable-diffusion-v1-5` 社区组织镜像 |
| SD 2.0 | 2022-11-24 | Stability AI | 865M | **OpenCLIP ViT-H/14** | 512² / 768²（v-pred） | CreativeML OpenRAIL++-M | 引入 **v-prediction**、depth2img、4× 超分；LAION 用 NSFW 过滤，**人体和画风能力下降，社区反响冷淡** |
| SD 2.1 | 2022-12-07 | Stability AI | 865M | OpenCLIP ViT-H | 768² | OpenRAIL++ | 放宽过滤后重训，部分修复 2.0 的问题 |
| Stable unCLIP 2.1 | 2023-03 | Stability AI | — | — | 768² | OpenRAIL++ | 图像变体 |
| SDXL 0.9 | 2023-06-22 | Stability AI | U-Net 2.6B | CLIP ViT-L + OpenCLIP ViT-bigG | 1024² | 研究许可 | 预览版 |
| **SDXL 1.0** | **2023-07-26** | Stability AI | Base 3.5B（U-Net 2.6B），加 Refiner 共 6.6B | 双 CLIP | 1024² | **CreativeML OpenRAIL++-M** | 分辨率与美学大幅提升；**第二个长寿生态**（Pony、Illustrious、NoobAI 等衍生） |
| SD Turbo / **SDXL Turbo** | 2023-11-28 | Stability AI | 同 SD2.1 / SDXL | — | 512² | 非商用（后并入会员制） | **ADD（对抗扩散蒸馏）**，1–4 步实时出图 |
| Stable Cascade | 2024-02-12 | Stability AI | Stage C 3.6B（基于 Würstchen v3） | CLIP ViT-bigG | 1024² | 非商用 | 24 倍高压缩潜空间；短暂过渡后被 SD3 取代 |
| SD3（论文） | 2024-03-05 | Stability AI | 800M–8B 一系列 | CLIP-L + CLIP-G + T5-XXL | 1024² | — | 论文《Scaling Rectified Flow Transformers》：**MMDiT + Rectified Flow** |
| SD3 API | 2024-04-17 | Stability AI | 8B | — | — | API | |
| **SD3 Medium** | **2024-06-12** | Stability AI | **2B** MMDiT | 三编码器 | 1024² | 最初为限制性很强的非商用许可，2024-07 改为 **Stability AI Community License** | **"草地上躺着的女人"人体崩坏事件**和许可证争议使口碑受挫；Civitai 一度暂停 SD3 内容 |
| **SD 3.5 Large / Large Turbo** | **2024-10-22** | Stability AI | **8.1B** MMDiT | 三编码器 | 1MP | Community License（年收入低于 100 万美元免费商用） | 加入 QK-Norm 提升训练稳定性；Turbo 为 4 步 ADD 蒸馏 |
| SD 3.5 Medium | 2024-10-29 | Stability AI | 2.5B **MMDiT-X** | 三编码器 | 0.25–2MP | Community License | 适配消费级显卡 |
| SD 3.5 ControlNets | 2024-11 | Stability AI | — | — | — | Community License | Canny / Depth / Blur |
| （此后） | 2025–2026 | Stability AI | — | — | — | — | 只发布了 TensorRT、AMD GPU/NPU 优化版、3D/4D 等衍生，**没有新的开放文生图基座**。截至 2026-07，官方新闻页无 SD4 公告 |

### 3.2 Stability AI 公司层面的关键事件

- **2022-10**：完成 1.01 亿美元种子轮，估值约 10 亿美元。
- **2023**：现金流紧张，核心人员陆续流失。
- **2024-03**：LDM 原作者 Robin Rombach 等研究骨干离职；创始人 CEO Emad Mostaque 于 2024-03-23 辞职。
- **2024-06**：前 Weta Digital CEO Prem Akkaraju 出任 CEO，Sean Parker 等注资重组；2024-09 James Cameron 加入董事会。
- **2024-07**：推出 Stability AI Community License，作为 SD3 系列统一许可。
- **2025–2026**：转向企业 / 行业合作（WPP、EA 等）、音频（Stable Audio）和 3D。2026 年有报道称其完成新一轮融资（投资方包括 EA、索尼音乐、环球、华纳音乐、AMD Ventures 等），具体金额各来源说法不一。

> **启示**：SD 的历史价值在于"把生图模型开放给所有人"，并奠定了 LDM、ControlNet / LoRA 生态、WebUI / ComfyUI 工具链这套基础设施。但在模型能力的竞赛中，Stability 自 2024 年起已被 BFL 和中国机构超越。

---

## 4. 开源文生图模型发布全景（按阶段）

> 说明：日期优先采用官方公告或 Hugging Face 仓库的公开时间（HF 仓库创建日期可能比正式公告早几天，相差较大时以公告为准）。"许可证"取自 HF 模型卡元数据。完整表格见 CSV。

### 4.1 第一阶段：SD 一家独大与 U-Net 时代（2022.08 – 2023）

| 发布时间 | 模型 | 机构 | 规模 | 架构 / 特点 | 许可证 |
|---|---|---|---|---|---|
| 2022-08 | Stable Diffusion 1.x | CompVis / Stability / Runway | 0.86B | LDM U-Net + CLIP | OpenRAIL-M |
| 2022-11 | SD 2.x | Stability AI | 0.87B | OpenCLIP、v-pred | OpenRAIL++ |
| 2023-04 | Kandinsky 2.1 → 2.2（2023-07） | Sber AI（俄） | 约 3.3B | unCLIP 式图像先验，多语言 | Apache-2.0 |
| 2023-04-28 | **DeepFloyd IF** | DeepFloyd（Stability 资助） | 4.3B（Stage I XL）+ 级联超分 | **像素空间级联扩散 + T5-XXL**，首个能较好渲染文字的开源模型 | DeepFloyd IF License（非商用） |
| 2023-07-26 | **SDXL 1.0** | Stability AI | 3.5B / 6.6B | 双 CLIP、Refiner | OpenRAIL++ |
| 2023-09 | Würstchen v2 | Pernias 等（后并入 Stability） | 约 1B | 42 倍高压缩潜空间 | MIT |
| 2023-10 | LCM（潜在一致性模型） | 清华 | — | 2–4 步采样蒸馏 | MIT |
| 2023-11 | **PixArt-α** | 华为诺亚 / 港大等 | 0.6B | **DiT + T5**，训练成本约为 SD1.5 的 10% | OpenRAIL++ |
| 2023-11-28 | SDXL Turbo | Stability AI | — | ADD 蒸馏 | 非商用 |
| 2023-12 | Playground v2 → **v2.5**（2024-02） | Playground AI | SDXL 架构 | 美学优化，EDM 噪声调度 | Playground Community License |
| 2023-12 | Kandinsky 3.0 | Sber AI | 约 12B（含 Flan-UL2 编码器） | U-Net + 大型文本编码器 | Apache-2.0 |

### 4.2 第二阶段：DiT 与 Flow Matching 转型，FLUX 接棒（2024）

| 发布时间 | 模型 | 机构 | 规模 | 架构 / 特点 | 许可证 |
|---|---|---|---|---|---|
| 2024-02-12 | Stable Cascade | Stability AI | 3.6B | Würstchen v3 | 非商用 |
| 2024-04 | **PixArt-Σ** | 华为诺亚等 | 0.6B | 弱到强训练，4K | OpenRAIL++ |
| 2024-04/05 | Lumina-T2X / Lumina-Next（2024-06） | 上海 AI Lab | 2B–5B | Flag-DiT，多模态 | Apache-2.0 |
| 2024-05-14 | **Hunyuan-DiT** | 腾讯 | 1.5B | 中英双语 DiT（CLIP + mT5） | Tencent Hunyuan Community |
| 2024-06-12 | SD3 Medium | Stability AI | 2B | MMDiT + RF | SAI 非商用 → Community |
| 2024-07-06 | **Kolors（可图）** | 快手 | U-Net 约 2.6B | **ChatGLM3-6B 作文本编码器**，中文能力强 | Apache-2.0（商用需登记） |
| 2024-07-12 | **AuraFlow v0.1–0.3** | fal.ai | 6.8B | MMDiT，号称"最大的真·开源（Apache）"模型 | Apache-2.0 |
| **2024-08-01** | **FLUX.1 [dev] / [schnell]**（[pro] 走 API） | **Black Forest Labs** | **12B** | **MMDiT 与单流 DiT 混合 + Rectified Flow**，CLIP-L + T5-XXL；dev 为指导蒸馏，schnell 为 1–4 步蒸馏 | dev：FLUX.1 非商用；**schnell：Apache-2.0** |
| 2024-10 | CogView3-Plus | 智谱 | 3B | DiT | Apache-2.0 |
| 2024-10 | OmniGen v1 | 智源 BAAI | 3.8B | 统一生成与编辑（基于 Phi-3） | MIT |
| 2024-10 | Janus（2024-10）/ JanusFlow（2024-11） | DeepSeek | 1.3B | 自回归统一理解与生成 | MIT |
| 2024-10-22 | **SD 3.5 Large / Turbo**；Medium（10-29） | Stability AI | 8.1B / 2.5B | MMDiT（-X） | Community |
| 2024-11 | **Sana** 0.6B / 1.6B | NVIDIA / MIT / 清华 | 0.6–1.6B | **线性注意力 DiT + DC-AE（f32 深压缩）+ Gemma-2 编码器**，4K、笔记本可跑 | Apache-2.0（代码）/ NSCL → 后改 Apache |
| 2024-11-21 | FLUX.1 Tools（Fill / Depth / Canny / Redux） | BFL | 12B | 修补、结构控制、图像变体 | 非商用 |

### 4.3 第三阶段：大模型化、编辑一体化、中国机构集中爆发（2025）

| 发布时间 | 模型 | 机构 | 规模 | 架构 / 特点 | 许可证 |
|---|---|---|---|---|---|
| 2025-01 | **Lumina-Image 2.0** | 上海 AI Lab | 2.6B | Unified Next-DiT + Gemma-2-2B | Apache-2.0 |
| 2025-01-27 | **Janus-Pro 1B / 7B** | DeepSeek | 7B | 自回归统一模型（解耦视觉编码） | MIT |
| 2025-01 起 | Chroma → Chroma1-HD（2025-08） | lodestones（社区） | 8.9B | 基于 FLUX.1 schnell 改造重训，社区自筹算力 | Apache-2.0 |
| 2025-03-04 | **CogView4-6B** | 智谱 | 6B | DiT + GLM-4-9B 编码器，原生中文 | **Apache-2.0** |
| 2025-03 | Sana 1.5 / Sana-Sprint | NVIDIA | 1.6–4.8B | 深度扩展 + 1–4 步 sCM 蒸馏 | Apache-2.0 |
| 2025-04-07 | **HiDream-I1**（Full / Dev / Fast） | HiDream（智象未来） | **17B** | **稀疏 MoE DiT**，Llama-3.1-8B + T5 + 双 CLIP | **MIT** |
| 2025-04 | Step1X-Edit | 阶跃星辰 | 约 19B | MLLM + DiT 指令编辑 | Apache-2.0 |
| 2025-04 | HiDream-E1 | HiDream | 17B | 指令编辑 | MIT |
| 2025-05 | **BAGEL-7B-MoT** | 字节 Seed | 7B 激活 / 14B 总 | **MoT 统一多模态**（理解、生成、编辑） | Apache-2.0 |
| 2025-05-29（API）/ **06-26（dev 权重）** | **FLUX.1 Kontext [dev]** | BFL | 12B | **上下文图像编辑**（文本加参考图，角色一致性） | 非商用 |
| 2025-06 | OmniGen2 | 智源 BAAI | 约 7B | 双路径（VLM + 扩散）统一生成与编辑 | Apache-2.0 |
| 2025-07-31 | FLUX.1 Krea [dev] | BFL × Krea | 12B | "去 AI 味"美学，写实风格 | 非商用 |
| **2025-08-04** | **Qwen-Image** | **阿里 Qwen** | **20B** | **MMDiT + Qwen2.5-VL-7B 编码器**，中英文长文本渲染突破 | **Apache-2.0** |
| 2025-08-18 | Qwen-Image-Edit → Edit-2509（多图）→ Edit-2511 | 阿里 Qwen | 20B | 语义与外观双编辑，多参考图 | Apache-2.0 |
| 2025-08 | NextStep-1 | 阶跃星辰 | 14B | **连续 token 的自回归**生成 | Apache-2.0 |
| 2025-09-09 | HunyuanImage 2.1 | 腾讯 | 17B | 原生 2K、双文本编码器、精炼模型 | Tencent Community |
| 2025-09 | SRPO | 腾讯 | （FLUX.1 dev 微调） | 在线 RL 对齐，提升写实度 | — |
| **2025-09-28** | **HunyuanImage 3.0** | 腾讯 | **80B MoE（64 专家，激活 13B）** | **原生多模态自回归框架**（基于 Hunyuan-A13B），当时最大的开源生图模型 | Tencent Community（排除欧盟、英国、韩国） |
| 2025-10 | FIBO | Bria AI | 8B | **结构化 JSON 提示**，授权数据训练 | CC BY-NC 4.0 |
| 2025-11 | Kandinsky 5.0 Image Lite | Sber（kandinskylab） | 6B | 文生图与图生图 | MIT |
| **2025-11-25** | **FLUX.2 [dev]**（[pro] / [flex] 走 API）+ FLUX.2 VAE | BFL | **32B** | Latent Flow Matching + **Mistral Small 3.1（24B）VLM 编码器**，最多 10 张参考图，4MP | dev：非商用；**VAE：Apache-2.0** |
| **2025-11-27** | **Z-Image-Turbo** | **阿里通义 Tongyi-MAI** | **6B** | **S3-DiT 单流 DiT** + Qwen3-4B 编码器，8 步、16GB 显存，中英双语 | **Apache-2.0** |
| 2025-12-04 | **LongCat-Image** / Edit | 美团 | 6B | 中文字符覆盖率高、写实 | Apache-2.0 |
| 2025-12 | Qwen-Image-Layered | 阿里 Qwen | 20B | 把图像拆成多个 RGBA 图层 | Apache-2.0 |
| 2025-12 | NewBie-image Exp0.1 | NewBie-AI（社区） | — | 二次元方向 | 自定义 |
| 2025-12-31 | **Qwen-Image-2512** | 阿里 Qwen | 20B | 写实度与文字进一步提升 | Apache-2.0 |

### 4.4 第四阶段：小模型回归、统一多模态、许可证分化（2026）

| 发布时间 | 模型 | 机构 | 规模 | 架构 / 特点 | 许可证 |
|---|---|---|---|---|---|
| 2026-01-14 | **GLM-Image** | 智谱 Z.ai | **9B AR + 7B DiT 解码器** | **自回归与扩散混合**（AR 基于 GLM-4-9B，加带 Glyph Encoder 的单流 DiT），知识密集型和文字场景强 | MIT |
| 2026-01-15 | **FLUX.2 [klein] 4B / 9B**（含 Base 版） | BFL | 4B / 9B | 统一生成与编辑，少步蒸馏，约 13GB 显存；2026-03 追加 9B-KV 版 | **4B：Apache-2.0**；9B：非商用 |
| 2026-01-26 | HunyuanImage-3.0-Instruct（含 Distil） | 腾讯 | 80B MoE | 带推理（thinking）的指令编辑 | Tencent Community |
| 2026-01-28 | **Z-Image（Base，非蒸馏）** | 阿里通义 | 6B | 支持 CFG、负面提示，便于微调和 LoRA 训练 | Apache-2.0 |
| 2026-02 | LongCat-Image-Edit-Turbo | 美团 | 6B | 蒸馏编辑 | Apache-2.0 |
| 2026-04 | **ERNIE-Image / -Turbo** | 百度 | **8B** 单流 DiT | 带 Prompt Enhancer；海报、漫画等结构化排版强；Turbo 为 8 步（DMD + RL）；24GB 显存 | **Apache-2.0** |
| 2026-04-22 | SenseNova-U1-8B-MoT | 商汤 | 8B | **NEO-unify 原生统一架构**，之后陆续推出信息图专版 | Apache-2.0 |
| 2026-04 | LLaDA2.0-Uni | 蚂蚁 inclusionAI | — | 离散扩散语言模型统一多模态 | Apache-2.0 |
| **2026-05-08** | **HiDream-O1-Image**（及 Dev / Dev-2604） | HiDream | **8B** | **像素级统一 Transformer（UiT）：无 VAE、无独立文本编码器**，原生 2048²，自带推理型 Prompt Agent | **MIT** |
| **2026-05-31** | **Cosmos3-Super-Text2Image** | NVIDIA | **64B** | **MoT：自回归塔 + 扩散塔**，有"agentic"模式；07 月出 4 步蒸馏版 | OpenMDW 1.1 |
| **2026-06-03** | **Ideogram 4.0**（开放权重，NF4 量化发布） | Ideogram | 约 9.3B | Flow-Matching DiT，结构化 JSON 提示加边界框排版 | **Ideogram 4 Non-Commercial**（代码 Apache-2.0；商用需付费许可。**注意：部分媒体误报为 Apache-2.0**） |
| 2026-06（约 06-18～22） | **Krea 2 Raw / Turbo** | Krea AI | 12.8B | Raw 便于微调，Turbo 为 8 步蒸馏 | Krea 2 Community License |
| 2026-07 | Fibo 1.5 / Fibo-Edit 1.5（08 月） | Bria AI | 8B | JSON 原生 | 自定义（非商用） |
| 2026-07-22 | **Mage-Flow**（Base / RL / Turbo / Edit） | 微软亚洲研究院 | **4B** | Mage-VAE + NR-MMDiT，原生分辨率；Turbo 在 A100 上 0.59 秒出一张 1024 图；自报 GenEval 0.90 | MIT（"仅供研究"）；**官方仓库后来无法访问，社区有镜像** |
| 2026-07-24 | FLUX.3 系列发布（Video / Action 抢先体验） | BFL | — | 多模态（视频、图像、机器人） | 闭源 / API |
| 2026-08-19 | SenseNova-U1.5-8B-MoT | 商汤 | 8B MoT（总约 17.5B） | 原生 4K 输出 | Apache-2.0 |
| 2026-09-04 | **LLaDA-Image / -Turbo** | 蚂蚁 inclusionAI | 6B | 离散扩散 LLM（LLaDA2）生成 VQ token，再交给 DiT；Turbo 为 4 步 | Apache-2.0 |
| **2026-09（中旬）** | **Qwen-Image-2.1**（10 月追加 Turbo） | 阿里 Qwen | **视觉生成部分 7B**（32 层单流 DiT） | 生成与编辑统一，**原生 RGBA 透明图**，最多 10 张参考图；多份第三方榜单称其为当时开放权重第一 | **Qwen Research License（非商用）**。**这是 Qwen-Image 系列首次放弃 Apache-2.0** |
| 2026-09-17 | Ming-Image-0.1-Design | 蚂蚁 inclusionAI | 6B | 专攻 UI、信息图、海报等富文本设计，RGBA | MIT |
| 2026-10-11 | MOSS-Image（Base / SFT） | OpenMOSS（复旦系） | — | 社区已基于它训练衍生模型（如 Kromoss） | Apache-2.0 |
| **2026-10-15** | **FLUX.3 Image Klein**（含 Base） | BFL | **5.6B** | 从 FLUX.3 蒸馏，原生 4K，最多 10 张参考图；INT8 量化后 8GB 显存可跑 | **Flux Kommunity License**（自定义） |
| 2026-10-18 | Fibo 2.0 dev / Edit-Turbo | Bria AI | 8.1B | 编辑、修补、多参考 | 非商用（CC BY-NC 系） |

> **待确认 / 未发布**：FLUX 3 Image 主模型目前只走 API（2026-10-01 起）；FLUX 3 Dev 开放权重已承诺但尚未发布；Ideogram 4.5（2026-09-30）未开放权重；Z-Image-Edit 的状态各来源说法不一。

### 4.5 参数规模演变一览

```
SD 1.5 (2022)        ▏0.86B
SDXL (2023)          ▎2.6B (UNet) / 3.5B
SD3 Medium (2024)    ▎2B
SD3.5 Large (2024)   ▊8.1B
FLUX.1 (2024)        █▏12B
HiDream-I1 (2025)    █▋17B
Qwen-Image (2025)    ██ 20B
FLUX.2 dev (2025)    ███▏32B
Cosmos3-Super (2026) ██████▍64B
HunyuanImage3 (2025) ████████ 80B MoE（激活 13B）
------ 2026 年回落到高效区间 ------
Mage-Flow 4B | FLUX.2 klein 4B/9B | FLUX.3 Klein 5.6B | Z-Image 6B | LongCat 6B
Qwen-Image-2.1 7B | ERNIE-Image 8B | HiDream-O1 8B | SenseNova-U1.5 8B
```

---

## 5. 技术演进主线

### 5.1 去噪骨干：U-Net → DiT → MMDiT → 单流 / 统一 Transformer

| 阶段 | 代表 | 说明 |
|---|---|---|
| U-Net + 交叉注意力 | SD1.x/2.x、SDXL、Kolors、Playground | 卷积下采样 / 上采样，文本经交叉注意力注入 |
| DiT（Peebles & Xie，2022.12） | PixArt-α/Σ、Hunyuan-DiT、Lumina | 用 Transformer 取代 U-Net，扩展规律更清晰（Scaling Law） |
| **MMDiT**（双流） | SD3/3.5、AuraFlow、Qwen-Image | 文本和图像各有一套权重，在注意力中拼接，信息双向流动 |
| 双流 + 单流混合 | FLUX.1 / FLUX.2 | 前若干层双流，后续层合并成单流，兼顾对齐与参数效率 |
| **单流 DiT**（S3-DiT 等） | Z-Image、ERNIE-Image、Qwen-Image-2.1、GLM-Image 解码器 | 文本 token 与图像 token 共享一套权重，参数效率更高，是 2026 年小模型的主流 |
| 稀疏 MoE DiT | HiDream-I1 | 用 MoE 扩容 |
| **自回归 / 统一多模态** | Janus-Pro、BAGEL、HunyuanImage 3.0、NextStep-1、GLM-Image、Cosmos3（MoT）、SenseNova-U1 | 把 LLM 的世界知识和推理能力带入生图，"先想后画" |
| **像素级、无 VAE** | HiDream-O1-Image（UiT） | 去掉 VAE 和独立文本编码器，端到端在像素上建模 |
| 离散扩散 LLM 加 DiT | LLaDA-Image | 用扩散式语言模型取代自回归 LLM 做语义规划 |

### 5.2 训练目标与采样

- **ε 预测（DDPM）** → **v 预测**（SD 2.x，改善高分辨率和零终端信噪比问题）→ **EDM 噪声调度**（Playground v2.5）→ **Rectified Flow / Flow Matching**（SD3、FLUX、之后几乎所有模型）。
- SD3 论文的贡献：系统比较各种噪声调度，提出 logit-normal 时间步采样，并验证 RF 在大规模下的优势。
- 求解器：DDIM → DPM-Solver / DPM-Solver++（2022）→ UniPC → 各类 flow ODE 求解器。

### 5.3 加速与蒸馏（步数从 50 降到 1–8）

| 技术 | 时间 | 代表 |
|---|---|---|
| Progressive Distillation | 2022 | — |
| **LCM / LCM-LoRA** | 2023.10–11 | 2–4 步 |
| **ADD（对抗扩散蒸馏）** | 2023.11 | SDXL Turbo、SD3.5 Large Turbo |
| SDXL-Lightning（渐进对抗蒸馏） | 2024.02 | 字节 |
| Hyper-SD、TCD、PCM | 2024 | — |
| 指导蒸馏 / 时间步蒸馏 | 2024.08 | FLUX.1 dev / schnell |
| LADD（潜在对抗蒸馏） | 2024 | SD3 Turbo |
| **DMD / DMD2** | 2023–2024 | ERNIE-Image-Turbo、Z-Image-Turbo 等 |
| sCM（连续时间一致性） | 2024.10 | Sana-Sprint |
| 2026 年现状 | — | **"Base 加 Turbo 双发布"已成惯例**：Base（28–50 步）用于微调，Turbo（4–8 步）用于推理 |

### 5.4 文本编码器：从 CLIP 到 LLM / VLM

| 阶段 | 编码器 | 代表模型 |
|---|---|---|
| CLIP | CLIP ViT-L/14（123M） | SD1.x |
| 更大的 CLIP | OpenCLIP ViT-H、ViT-bigG | SD2.x、SDXL |
| T5 | T5-XXL（4.7B） | Imagen、DeepFloyd IF、PixArt、SD3、FLUX.1 |
| 中文 LLM | ChatGLM3-6B、mT5、GLM-4-9B | Kolors、Hunyuan-DiT、CogView4 |
| 小型 LLM | Gemma-2-2B、Llama-3.1-8B、Qwen3-4B | Sana、Lumina 2、HiDream-I1、Z-Image |
| **VLM** | Qwen2.5-VL-7B、Mistral Small 3.1 24B | Qwen-Image、FLUX.2（让编辑和多参考图天然统一） |
| 一体化 | 统一 Transformer 本身 | HunyuanImage 3.0、HiDream-O1、Cosmos3 |

**影响**：复杂提示遵循、空间关系、计数、**文字渲染**（尤其中文）大幅提升。2023 年 DALL·E 3 提出的"合成长描述重标注"（recaptioning）也被开源界普遍采用；2025–2026 年再加上"Prompt Enhancer / Prompt Agent"（ERNIE、HiDream-O1、FLUX.3 Klein 都推荐使用）。

### 5.5 VAE / Tokenizer

- SD1.x–SDXL：f8、4 通道 KL-VAE，细节和文字是瓶颈。
- SD3 / FLUX.1：**16 通道 VAE**，细节和文字明显改善。
- Sana：DC-AE（f32 深度压缩），token 数减少约 16 倍。
- FLUX.2：新训练的 VAE 以 Apache-2.0 单独开源；2026-04 又推出 FLUX.2 small decoder，成为社区复用的基础组件（如 Kromoss）。
- Mage-Flow：Mage-VAE；HiDream-O1：**完全不用 VAE**。

### 5.6 可控性与个性化：从"外挂"到"原生"

1. **2022–2023 外挂期**：Textual Inversion（2022.08）、DreamBooth（2022.08）、**LoRA**（2023.01 起被移植到 SD）、**ControlNet**（2023.02，张吕敏）、T2I-Adapter、**IP-Adapter**（2023.08）、InstantID / PhotoMaker（2024.01）、PuLID。
2. **2024 过渡期**：FLUX.1 Tools（Fill / Canny / Depth / Redux）、OmniGen 统一指令。
3. **2025–2026 原生期**：**FLUX.1 Kontext、Qwen-Image-Edit、Step1X-Edit、BAGEL**，指令编辑和多参考图（最多 10 张）成为基座模型的原生能力；Qwen-Image-Layered / Qwen-Image-2.1 / Ming-Image 原生支持 RGBA 图层和透明背景。

### 5.7 对齐与后训练

- 美学微调（Emu 的 quality-tuning，2023）→ Diffusion-DPO（2023.11）→ ReFL / ImageReward → **Flow-GRPO、DanceGRPO（2025）** → SRPO（腾讯，2025.09）→ 2026 年 Turbo 模型普遍采用"蒸馏 + RL"（如 ERNIE-Image-Turbo：DMD + RL；Mage-Flow RL 版）。

---

## 6. 许可证演变

| 时期 | 代表许可证 | 典型模型 | 特点 |
|---|---|---|---|
| 2022 | **CreativeML OpenRAIL-M** | SD1.x | 允许商用，附"负责任使用"限制条款 |
| 2022–2023 | CreativeML OpenRAIL++-M | SD2.x、SDXL、PixArt | 同上 |
| 2023–2024 | 非商用研究许可 | SDXL Turbo、Stable Cascade、SD3 Medium 初版、DeepFloyd IF | Stability 尝试变现，引发社区反弹 |
| 2024.07– | **Stability AI Community License** | SD3/3.5 | 年收入低于 100 万美元免费商用 |
| 2024.08– | **BFL 双轨制** | FLUX.1 schnell（Apache）与 dev（非商用）；FLUX.2 klein 4B（Apache）与 9B / dev（非商用） | 小模型宽松、大模型收费，商业授权另售 |
| 2024–2026 | **Apache-2.0 / MIT**（中国机构为主） | CogView4、HiDream-I1/O1、Qwen-Image（≤2512）、Z-Image、LongCat、ERNIE-Image、SenseNova、LLaDA-Image、Ming-Image、GLM-Image | 真正可商用，是社区采用率高的重要原因 |
| 区域限制型 | Tencent Hunyuan Community | Hunyuan-DiT、HunyuanImage 2.1/3.0 | 排除欧盟、英国、韩国；月活超 1 亿需另行授权 |
| **2026 收紧潮** | Qwen Research、Ideogram 4 Non-Commercial、Krea 2 Community、Flux Kommunity、Bria FIBO（CC BY-NC）、NVIDIA OpenMDW | Qwen-Image-2.1、Ideogram 4、Krea 2、FLUX.3 Klein、Fibo 2.0、Cosmos3 | "开放权重"不等于"开源"，商用前**必须逐一核对 LICENSE** |

---

## 7. 闭源模型对照时间线

| 时间 | 闭源模型 | 同期开源对照 |
|---|---|---|
| 2022.04 | DALL·E 2 | — |
| 2022.05 | Imagen | — |
| 2022.07 | Midjourney 公测（v3） | SD 1.4（08 月） |
| 2022.11 | Midjourney v4 | SD 2.0 |
| 2023.03 | Midjourney v5；Adobe Firefly | ControlNet 生态 |
| 2023.08 | Ideogram 0.1（擅长文字） | SDXL 1.0 |
| 2023.10 | **DALL·E 3**（重标注 + ChatGPT 集成） | PixArt-α |
| 2023.12 | Midjourney v6；Imagen 2 | Playground v2 |
| 2024.08 | Ideogram 2.0；Imagen 3；Grok 用上 FLUX | FLUX.1 |
| 2024.10 | FLUX1.1 [pro]；Recraft V3 | SD 3.5 |
| 2025.03 | **GPT-4o 原生生图**（自回归，"吉卜力风"刷屏）；Ideogram 3.0 | CogView4 |
| 2025.04 | gpt-image-1 API；**Midjourney v7**；Seedream 3.0 | HiDream-I1 |
| 2025.05 | Imagen 4；FLUX.1 Kontext [pro/max] | BAGEL |
| 2025.08 | **Gemini 2.5 Flash Image（Nano Banana）** | Qwen-Image |
| 2025.09 | Seedream 4.0 | HunyuanImage 3.0 |
| 2025.11 | **Nano Banana Pro**（Gemini 3 Pro Image）；FLUX.2 [pro/flex] | FLUX.2 dev、Z-Image-Turbo |
| 2025.12 | GPT Image 1.5；FLUX.2 [max] | Qwen-Image-2512 |
| 2026.02 | Nano Banana 2（Gemini 3.1 Flash Image）* | FLUX.2 klein |
| 2026.03–07 | Midjourney V8 alpha（03）→ V8.1 → V8.2（07）* | — |
| 2026.04 | **GPT Image 2** * | ERNIE-Image |
| 2026.07 | Seedream 5.0 Pro *；FLUX.3（视频 / 动作） | Mage-Flow |
| 2026.09 | GPT Image 2.5（Flare / Sunburst）*；Ideogram 4.5 | Qwen-Image-2.1 |
| 2026.10 | Nano Banana 2.1 GA *；FLUX 3 Image（API） | FLUX.3 Image Klein |

\* 标星的 2026 年闭源条目主要来自第三方报道和聚合站，日期可能有出入。

**差距观察**：多个 2026 年下半年的 Artificial Analysis 竞技场快照显示，闭源榜首（GPT Image 2 / 2.5）领先开放权重榜首约 100–160 Elo。开放权重的头部名单在 Qwen-Image-2.1、Cosmos3-Super（agentic）、HiDream-O1-Image-Dev-2604、Ideogram 4、FLUX.2 dev、ERNIE-Image 之间轮换，不同快照排名不一致，以实时榜单为准。

---

## 8. 社区生态与工具链

| 类别 | 代表 | 时间 | 说明 |
|---|---|---|---|
| 推理库 | **HF diffusers** | 2022.07– | 几乎所有新模型首日支持的标准库 |
| WebUI | **AUTOMATIC1111 SD WebUI** | 2022.08– | SD1.5 / SDXL 时代的事实标准，2024 年后更新放缓 |
| 节点式工作流 | **ComfyUI** | 2023.01– | 2024 年后成为新模型首发的第一平台，现已公司化（Comfy Org） |
| 其他前端 | InvokeAI、Fooocus（2023.08）、SD.Next、Forge / Forge Neo、SwarmUI | — | — |
| 模型社区 | **Civitai**（2022.11–）、HF Hub、LiblibAI / 吐司（国内） | — | LoRA 和微调模型分发 |
| 微调工具 | kohya-ss sd-scripts、OneTrainer、ai-toolkit、DiffSynth-Studio（ModelScope） | — | LoRA / 全量微调 |
| 加速推理 | TensorRT、Nunchaku（SVDQuant 4-bit）、GGUF 量化、NVFP4（Blackwell）、SGLang-Diffusion、vLLM-Omni | 2024–2026 | 2026 年官方常同时发布 FP8 / NVFP4 权重 |
| 现象级社区模型 | NovelAI 泄露模型（2022.10）、Anything、Pony Diffusion V6 XL（2024.01）、Illustrious XL（2024.09）、NoobAI-XL、Chroma、Kromoss | — | 二次元 / 风格化方向长期由社区主导 |

**社区迁移路径**：SD1.5 → SDXL（2023–2024 年两代并存）→ FLUX.1 dev（2024 下半年起）→ 2025 年分流到 Qwen-Image / HiDream / Chroma → **2025 年末起 Z-Image-Turbo 凭借 6B、Apache 许可和 16GB 显存成为新的大众基座** → 2026 年 FLUX.2 klein 4B、Z-Image Base、ERNIE-Image、Qwen-Image-2.1、FLUX.3 Klein 并存。

---

## 9. 数据、版权与监管

- **LAION-5B**（2022.03）：SD 1.x/2.x 的数据来源。2023-12 斯坦福互联网观察站在其中发现 CSAM 链接，LAION 下架数据集，2024-08 发布清理后的 **Re-LAION-5B**。
- **诉讼**：
  - *Andersen v. Stability AI 等*（美国，2023.01 起，艺术家集体诉讼）：部分诉求进入证据开示阶段。
  - *Getty Images v. Stability AI*：美国案件进行中；**英国高等法院于 2025-11 判决**，认定模型权重本身不构成"侵权复制品"，Getty 的二次侵权主张大部分败诉，仅在旧版本生成带水印图像的商标问题上获部分支持。
- **授权数据路线**：Adobe Firefly、Bria FIBO / Fibo 2.0（授权数据训练）、Stable Audio 3.0。
- **监管**：欧盟 AI Act 的通用 AI 模型透明度义务分阶段生效（2025-08 起），这也是部分模型（如 Tencent Community License）在许可中排除欧盟的原因之一。中国《生成式人工智能服务管理暂行办法》（2023-08）和《人工智能生成合成内容标识办法》（2025-09 施行）要求对生成内容加显式或隐式标识。
- **安全**：Stable Diffusion 早期的 NSFW 过滤器、C2PA 内容凭证、不可见水印（如 SynthID）逐渐成为闭源模型标配；开源模型主要依赖许可证的使用政策约束。

---

## 10. 格局与趋势判断

1. **开源的主力从 Stability AI 转向 BFL 加中国大厂和独角兽，再加 NVIDIA**。Stability 在 2024-10 之后已无新的开放文生图基座。
2. **"Base + Turbo + Edit"三件套成为发布惯例**，并同时提供 FP8 / NVFP4 / GGUF 等量化版本，ComfyUI 和 diffusers 首日适配。
3. **参数量先膨胀后回落**：2025 年是 20B–80B 的"大模型年"；2026 年的主流开放模型回到 4–9B，靠更好的数据、单流架构、VLM 编码器和 RL 后训练取胜，目标是 8–16GB 消费级显卡。
4. **生成与理解融合**：HunyuanImage 3.0、GLM-Image、Cosmos3、SenseNova-U1、HiDream-O1、LLaDA-Image 代表的"统一模型 + 推理 + Prompt Agent"路线，正在缩小与 GPT Image / Nano Banana 这类原生多模态闭源模型的差距。
5. **文字渲染与设计场景成为新战场**：Qwen-Image（中文长文本）、ERNIE-Image（海报、漫画）、Ming-Image-Design（UI、信息图）、Ideogram 4（JSON 加边界框）、RGBA 透明图层（Qwen-Image-2.1 / Layered、Ming-Image）。
6. **许可证分化加剧**：一边是 Apache / MIT 持续供给（百度、通义 Z-Image、HiDream、商汤、蚂蚁），一边是"开放权重但非商用"的模式变多（Qwen-Image-2.1、Ideogram 4、Krea 2、FLUX.3 Klein）。商用落地要格外注意。
7. **视频与图像同源**：Wan、HunyuanVideo、LTX-2、FLUX.3 Video、Kandinsky 6 等视频模型与图像模型共享 VAE 和 DiT，图像模型越来越多地作为视频模型的子能力发布。

---

## 11. 选型建议（截至 2026-11）

| 需求 | 推荐（开放权重） | 许可提示 |
|---|---|---|
| 可商用、综合质量高、中文好 | **Z-Image（Turbo / Base）**、**ERNIE-Image**、Qwen-Image-2512 | Apache-2.0 |
| 极致画质，不在意商用 | Qwen-Image-2.1、FLUX.2 dev、Ideogram 4、HiDream-O1（MIT，可商用） | 前三者为非商用或研究许可 |
| 低显存 / 实时 | FLUX.2 klein 4B（Apache）、FLUX.3 Image Klein（8GB）、Mage-Flow Turbo、Sana-Sprint | 注意各自许可 |
| 图像编辑 / 多参考图 | Qwen-Image-Edit-2511（Apache）、FLUX.2 klein、FLUX.1 Kontext dev、LongCat-Image-Edit | — |
| 文字、海报、UI 设计 | ERNIE-Image、Ming-Image-0.1-Design（MIT）、GLM-Image（MIT）、Qwen-Image 系列 | — |
| 二次元 / 风格化社区生态 | SDXL 系（Illustrious / NoobAI）、Chroma、NewBie-image | 依具体模型 |
| 学习 / 教学 / 研究基线 | SD 1.5 → SDXL → SD 3.5 Medium → FLUX.1 schnell，加上 diffusers 源码 | 适合理解演进脉络 |

---

## 12. 方法说明与信息源

**方法**

- 发布时间、许可证、参数量优先取自 **Hugging Face Hub API 和官方模型卡**（`/api/models?author=…`、`/api/models/{id}` 的 `cardData`、`safetensors.total`），在 2026-11-02 实时查询。
- 2022–2025 年的历史事实基于论文、官方博客和广泛报道；2026 年的条目以 HF 官方仓库为准，再用新闻和第三方榜单交叉核对。
- 来源冲突时（如 Ideogram 4 许可证、FLUX.3 首发日期、竞技场 Elo），正文已标注，并以官方仓库为准。

**主要信息源**

- 论文：DDPM（arXiv 2006.11239）、DDIM（2010.02502）、ADM（2105.05233）、CFG（2207.12598）、LDM（2112.10752）、DiT（2212.09748）、SDXL（2307.01952）、SD3 / Rectified Flow Transformers（2403.03206）、PixArt-α（2310.00426）、Sana（2410.10629）、ControlNet（2302.05543）、LCM（2310.04378）、ADD（2311.17042）、HiDream-O1 技术报告（2605.11061）
- 官方仓库：Hugging Face 上的 `CompVis`、`stabilityai`、`black-forest-labs`、`Qwen`、`Tongyi-MAI`、`tencent`、`baidu`、`zai-org`、`HiDream-ai`、`meituan-longcat`、`inclusionAI`、`sensenova`、`nvidia`、`ideogram-ai`、`krea`、`briaai`、`Efficient-Large-Model`、`Alpha-VLLM`、`deepseek-ai`、`ByteDance-Seed`、`stepfun-ai`、`Kwai-Kolors`、`PixArt-alpha`、`fal`、`lodestones`、`kandinskylab`、`DeepFloyd`、`OpenMOSS-Team`
- 百科与新闻：
  - [Wikipedia – Flux (text-to-image model)](https://en.wikipedia.org/wiki/Flux_(text-to-image_model))
  - [Wikipedia – Ideogram](https://en.wikipedia.org/wiki/Ideogram_(text-to-image_model))（其中 4.0 为 Apache-2.0 的说法与 HF 模型卡不符）
  - [Wikipedia – Midjourney](https://en.wikipedia.org/wiki/Midjourney)
  - [Pinggy – Best Open-Source AI Image Models to Self-Host in 2026](https://pinggy.io/blog/best_free_open_source_ai_image_generators_to_self_host/)
  - [ComfyUI Wiki – Z-Image-Base 发布](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release)
  - [ComfyUI Wiki – Mage-Flow](https://comfyui-wiki.com/en/news/2026-07-22-mage-flow-microsoft)
  - [Ideogram 商业许可页](https://ideogram.ai/licensing/)、[Ideogram Non-Commercial Model Agreement](https://ideogram.ai/legal/non-commercial-model-agreement/)
  - [Technobezz – FLUX 3 Image Klein](https://www.technobezz.com/news/black-forest-labs-flux-image-klein-parameters)、[DataNorth – FLUX 3](https://datanorth.ai/news/black-forest-labs-releases-flux-3)
  - [Artificial Analysis – Text to Image Arena](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)
  - [invideo – Open Source Image Models License Matrix (2026-08)](https://invideo.io/blog/open-source-image-models-licenses/)
  - [teamday.ai – Best AI Image Models Nov 2026](https://www.teamday.ai/blog/best-ai-image-models-2026)

> 局限：2026 年部分闭源模型的日期和竞技场 Elo 来自第三方聚合站；HF 仓库创建日期可能早于正式发布日期几天。建议在关键决策（尤其商用许可）前查看对应官方仓库的 LICENSE 文件。
