# 运行脚本说明

在你自己的机器上(Linux/macOS)装好 `colab` 命令行(`uv tool install google-colab-cli`)并登录后,所有 `vm_*.py` 都是用
`colab exec -s <会话名> -f <脚本>` 在 Colab 虚拟机里执行的;上传文件用 `colab upload`,取回用 `colab download`。

| 脚本 | 作用 |
|---|---|
| `vm_install.py` | 克隆 ComfyUI,后台 `pip install`(exec 一分钟左右会超时,所以必须放后台) |
| `vm_status.py` | 一次性查看:安装是否完成、Drive 是否挂载、ComfyUI 是否在线、最近任务结果 |
| `vm_start.py` | 启动 ComfyUI(模型读 Drive),并确认自定义节点已注册(重复运行安全) |
| `vm_generate.py` | 读 `/content/gen_jobs.json`,提交 Anima + LoRA 文生图任务 |
| `vm_repair.py` | 读 `/content/repair_jobs.json`,提交手/脚局部重绘(支持分段、`grid`、`params`、`batch`) |
| `vm_submit.py` | 提交一个现成的 API 格式工作流 `/content/test_wf.json` |
| `convert_inpaint_workflow.py` | 本机运行:把结果 PNG 里的工作流转成纯远端可跑的 JSON |
| `helpers/vm_queue.py` | 看队列和各任务已出图数量 |
| `helpers/vm_cancel.py` | 取消还没开始的任务(名字写在 `/content/cancel_names.json`) |
| `helpers/vm_progress_once.py` | 打印一行进度,配合本地 `for` 循环可持续看 |
| `helpers/vm_restart_comfyui.py` | 结束 ComfyUI 进程(之后再运行 `vm_start.py`,用于加载新节点代码) |
| `helpers/vm_save_to_drive.py` | 把出好的图存进 Drive 的新文件夹(先改里面的路径) |
| `helpers/test_segsinpaint_nodes.py` | 本地用假模型测节点逻辑(需要 torch) |
| `examples/*.json` | 出图 / 批量重绘 / 脚爪重绘的任务文件示例 |

典型顺序:`vm_install` → 上传 `colab/custom_nodes/ComfyUI-SEGSInpaint/` → `vm_start` → (上传任务 JSON)→ `vm_generate` 或 `vm_repair` → `vm_status`/`helpers/vm_queue` 看进度 → `colab download`。
**重绘前必须先得到使用者批准;看图一律直接打开原图,不用脚本。**
