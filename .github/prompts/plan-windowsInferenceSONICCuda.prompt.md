````markdown
# Plan A+：Windows 推理极速启动（SONIC + CUDA）✅（最快看到效果）

> 目标：在 Windows + NVIDIA GPU（如 RTX 3060Ti）上，用 **tinyworlds 的 SONIC 预训练模型** 跑通推理闭环：能加载数据与模型、GPU 参与计算、输出 PNG（可选 MP4）。

---

## 前置条件（必须）
- **NVIDIA 显卡驱动已安装**，并且命令行可用：
  - `nvidia-smi`
- Python：推荐 3.10 或 3.11
- 在仓库根目录执行所有命令（`AlmondGod/tinyworlds`）

> 说明：一般不需要单独安装 CUDA Toolkit。使用 PyTorch 的 CUDA wheel（`cu126/cu128/cu129`）即可获得 CUDA runtime。

---

## Step 0：快速自检（30 秒）
在 PowerShell 中运行：

```powershell
nvidia-smi
python --version
````

---

## Step 1：创建并激活虚拟环境（强烈建议）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

---

## Step 2：安装 GPU 版 PyTorch（锁定 CUDA wheel）

> 只选其中一条（通常先试 cu126；若失败再换 cu128/cu129）。

```powershell
pip uninstall torch torchvision torchaudio -y

# CUDA 12.6（推荐先试）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

（备选）

```powershell
# CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

```powershell
# CUDA 12.9
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

### 验证 GPU 可用（必须看到 True）

```powershell
python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.version.cuda,'is_available=',torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Step 3：安装其余依赖（Step 3A：剥离 torch，防止被覆盖）

> 目的：避免 `pip install -r requirements.txt` 把你已经装好的 GPU 版 torch 覆盖成 CPU 版/不匹配版。

```powershell
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio)\b' } | Set-Content requirements-notorch.txt
pip install -r requirements-notorch.txt
```

> 如果仓库不是 requirements.txt（而是 pyproject.toml/setup.py），原则不变：**先装 GPU torch，再装其它依赖**，并尽量避免后续步骤重新安装 torch。

---

## Step 4：下载 SONIC 数据 + 预训练模型（与仓库脚本一致）

先确认脚本支持的子命令：

```powershell
python scripts/download_assets.py --help
```

然后执行（命令按仓库实现：`datasets` / `models`）：

```powershell
# 1) 下载 SONIC 数据（只下需要的 h5，最快）
python scripts/download_assets.py datasets --pattern "sonic_frames.h5"

# 2) 下载 SONIC 预训练模型（tokenizer / latent_actions / dynamics）
python scripts/download_assets.py models --suite-name sonic
```

### 期望落地结果（至少满足其一）

* `data/` 下出现 `sonic_frames.h5`（或类似命名）
* 某个模型目录下出现 `.pth` 文件（HF 发布常见为单文件 .pth）
* 或出现目录式 checkpoint（包含 `model_state_dict.pt` / `state.pt`）

---

## Step 4.5：⚠️ 权重格式兼容性检查（只在报错时处理）

> 若推理报错类似：
>
> * `NotADirectoryError: ... .pth/model_state_dict.pt`
> * `FileNotFoundError: ... model_state_dict.pt` 或 `state.pt`
>
> 说明：代码加载器期待“目录式 checkpoint”，但下载到的是“单文件 .pth”。

### 最快修复策略（推荐交给 Copilot 直接改）

在以下函数里加一个“如果是 .pth 就按单文件加载”的分支：

* `utils/utils.py`：

  * `load_videotokenizer_from_checkpoint`
  * `load_latent_actions_from_checkpoint`
  * `load_dynamics_from_checkpoint`

分支逻辑参考（伪代码）：

```python
from pathlib import Path
import torch

p = Path(checkpoint_path)
if p.is_file() and p.suffix == ".pth":
    obj = torch.load(p, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    sd  = obj.get("model") or obj.get("state_dict") or obj.get("model_state_dict") or obj
    # 然后继续走 set_model_state_dict(...)
else:
    # 原逻辑：p / "model_state_dict.pt" + p / "state.pt"
    ...
```

---

## Step 5：运行推理（尽量少改 yaml）

### 5.1 设置 PYTHONPATH（Windows PowerShell）

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
```

### 5.2 查看推理脚本参数（建议）

```powershell
python scripts/run_inference.py --help
```

### 5.3 启动推理（推荐先关闭交互，最快出结果）

```powershell
python scripts/run_inference.py --config configs/inference.yaml -- device=cuda dataset=SONIC use_interactive_mode=false use_actions=true
```

> 说明：
>
> * `use_actions=true`：用 latent_actions（更接近作者演示的“动作驱动”）
> * `use_interactive_mode=false`：不需要每步输入 action id，最快看到生成结果

（可选）开启“逐步交互”（控制台输入 action id，回车一步一帧）：

```powershell
python scripts/run_inference.py --config configs/inference.yaml -- device=cuda dataset=SONIC use_interactive_mode=true use_actions=false
```

---

## Step 6：如何确认“真的跑起来了”（硬验证）

### 6.1 日志证据（必须）

终端里应出现类似信息：

* 成功加载 dataset（h5 路径）
* 成功加载模型/权重（checkpoint 或 .pth 路径）
* 推理循环开始运行

### 6.2 文件证据（必须）

看到生成结果（优先看 PNG，比 MP4 更可靠）：

* `inference_results/` 或 `results/...` 下出现 `*.png`
* `*.mp4` 可能因编码器/codec 失败，但不影响“推理跑通”的结论

### 6.3 GPU 证据（强烈建议）

另开一个 PowerShell：

```powershell
nvidia-smi -l 1
```

推理期间显存占用/利用率应有变化。

---

## 常见故障快速定位

* `torch.cuda.is_available()==False`

  * 说明装到 CPU torch 或被覆盖：重跑 Step 2，并坚持 Step 3A
* 找不到数据（`.h5` 不存在）

  * 重新跑 Step 4 的 datasets 下载，确认 `data/` 下文件名匹配配置
* 找不到 checkpoint / 权重加载失败

  * 多半是“目录式 vs 单文件 .pth”格式差异：按 Step 4.5 处理
* MP4 不可播放/写入失败

  * 常见 codec 问题：先看 PNG 是否生成；需要时再换编码方式

---

## Key Files（入口点）

* 推理脚本：`scripts/run_inference.py`
* 下载脚本：`scripts/download_assets.py`
* 推理配置：`configs/inference.yaml`
* 权重加载：`utils/utils.py`（若遇到 .pth / checkpoint 格式问题）
* 可视化检查（可选）：`scripts/visualize_batch.py`

---

## Decisions（此 Plan 的取舍）

* **最快闭环优先**：SONIC + 预训练权重（更可能一次跑通）
* **先锁定 GPU torch**：避免依赖安装把 torch 覆盖成 CPU/不匹配版本
* **少改配置**：优先命令行覆盖（降低 Windows 路径/缩进/版本差异问题）

```

::contentReference[oaicite:0]{index=0}
```
