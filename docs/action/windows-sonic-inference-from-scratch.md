# Windows SONIC 推理完整实现记录

**日期**: 2026-02-01  
**目标**: Windows + NVIDIA GPU (RTX 3060 Ti) 上快速启动 tinyworlds SONIC 数据集推理  
**最终状态**: ✅ 推理成功（自动 + 交互模式均可用）

---

## 执行过程

### Step 0: 硬件环境自检
- **显卡**: NVIDIA GeForce RTX 3060 Ti (8GB)
- **驱动版本**: 560.94 with CUDA 12.6
- **Python 版本**: 3.11.4
- **系统**: Windows (PowerShell)
- ✅ **验证**: `nvidia-smi` 正常，CUDA runtime 可用

### Step 1: 虚拟环境设置
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```
- ✅ 虚拟环境激活成功，pip 升级至 26.0

### Step 2: PyTorch CUDA 安装（关键）
```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```
- PyTorch 版本: **2.8.0+cu126**
- 总大小: ~2.9 GB
- ✅ `torch.cuda.is_available()` = **True**
- ✅ `torch.cuda.get_device_name(0)` = **NVIDIA GeForce RTX 3060 Ti**

### Step 3: 依赖安装（剥离 torch）
创建 `requirements-notorch.txt`，移除了 torch/torchvision/torchaudio 依赖，防止被覆盖：
```powershell
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio)\b' } | Set-Content requirements-notorch.txt
pip install -r requirements-notorch.txt
```
**安装包**: wandb, h5py, opencv-python, omegaconf, huggingface_hub 等

- ✅ 所有依赖安装成功

### Step 4: 数据 + 模型下载

#### 4.1 SONIC 数据集下载
```powershell
python -c "
import os
from huggingface_hub import hf_hub_download
os.makedirs('data', exist_ok=True)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
path = hf_hub_download(
    repo_id='AlmondGod/tinyworlds',
    filename='sonic_frames.h5',
    repo_type='dataset',
    local_dir='data',
    local_dir_use_symlinks=False,
)
print(f'Downloaded: {path}')
"
```
- **文件**: `data/sonic_frames.h5`
- **大小**: 249 MB
- ✅ 下载成功

#### 4.2 预训练模型下载
```powershell
python -c "
import os
from datetime import datetime
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
repo_id = 'AlmondGod/tinyworlds-models'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path('results') / f'{timestamp}_sonic'

models = [
    'sonic/sonic_video_tokenizer_step_27500_2025_09_17_06_20_26.pth',
    'sonic/sonic_latent_actions_step_2500_2025_09_17_06_50_59.pth',
    'sonic/sonic_dynamics_step_97500_2025_09_18_11_25_59.pth',
]

for model_file in models:
    model_type = model_file.split('_')[1]
    target_dir = results_dir / f'{model_type}/checkpoints'
    path = hf_hub_download(repo_id=repo_id, filename=model_file, repo_type='model',
                           local_dir=str(target_dir), local_dir_use_symlinks=False)
    print(f'Downloaded: {path}')
"
```
- **下载位置**: `results/20260201_175440_sonic/`
  - `video/checkpoints/sonic/sonic_video_tokenizer_*.pth` (19.5 MB)
  - `latent/checkpoints/sonic/sonic_latent_actions_*.pth` (19.8 MB)
  - `dynamics/checkpoints/sonic/sonic_dynamics_*.pth` (16.2 MB)
- ✅ 3 个模型文件下载成功

### Step 4.5: 权重加载兼容性修复

**问题**: 官方权重为单个 `.pth` 文件，但代码期望目录式 checkpoint（包含 `model_state_dict.pt` + `state.pt`）

**修复**: 修改 `utils/utils.py` 中的三个加载函数（`load_videotokenizer_from_checkpoint`, `load_latent_actions_from_checkpoint`, `load_dynamics_from_checkpoint`），添加条件分支：

```python
# Handle both directory-based and single .pth file checkpoints
p = Path(checkpoint_path)
if p.is_file() and p.suffix == '.pth':
    # Single .pth file: load directly as state dict
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    model_sd = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt
    state_cfg = {'config': {}} if not isinstance(ckpt, dict) or 'config' not in ckpt else {'config': ckpt.get('config', {})}
else:
    # Directory-based: load from subdirectories
    model_sd = torch.load(p / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state_cfg = torch.load(p / STATE, map_location='cpu', weights_only=False)
```

同时改进 `conditioning_dim` 推断逻辑以支持单文件模式。

- ✅ 兼容性修复完成

### Step 5: 推理配置调整

修改 `configs/inference.yaml`：
```yaml
dataset: SONIC           # 改从 PONG 到 SONIC
device: cuda             # 改从 mps 到 cuda（Windows 必需）
use_actions: true        # 启用 action 模型
use_interactive_mode: false  # 非交互（自动推理）
generation_steps: 10
context_window: 2
```

- ✅ 配置修改完成

### Step 6: 推理执行

#### 6.1 自动推理（随机 action）
```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
python scripts/run_inference.py --config configs/inference.yaml
```

**结果**:
- ✅ 成功加载所有模型和数据
- ✅ 生成 10 帧预测
- MSE (GT vs Pred): 0.078128
- 输出文件:
  - `inference_results/inference_results_gt_vs_pred_20260201_175658.png` (312 KB)
  - `inference_results/inference_video_20260201_175658.mp4` (21 KB)

#### 6.2 交互模式推理（用户输入 action）
```powershell
python scripts/run_inference.py --config configs/inference.yaml -- use_interactive_mode=true use_actions=false
```

**交互流程**:
```
Inferring frame 1/10
using interactive mode
Enter action id [0..15] for step 1: 5
Inferring frame 2/10
using interactive mode  
Enter action id [0..15] for step 2: 7
... (继续交互)
```

- ✅ 交互模式运行正常
- 可以依次输入 0-15 的 action 编号控制推理过程

---

## 关键修改文件

| 文件 | 修改内容 |
|------|--------|
| `utils/utils.py` | 支持单文件 .pth checkpoint 加载（3 个函数） |
| `configs/inference.yaml` | device: cuda, dataset: SONIC, use_actions: true |
| `requirements-notorch.txt` | 新建，移除 torch 依赖 |

---

## 快速重启命令

```powershell
# 激活环境 + 设置 PYTHONPATH + 自动推理
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
python scripts/run_inference.py --config configs/inference.yaml

# 或交互模式
python scripts/run_inference.py --config configs/inference.yaml -- use_interactive_mode=true use_actions=false
```

---

## 输出位置

所有推理结果保存到 `inference_results/` 目录：
- **PNG**: 真实 vs 生成帧对比图
- **MP4**: 完整视频序列

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| CUDA unavailable | PyTorch 装到 CPU 版本 | 重跑 Step 2，验证 `torch.cuda.is_available()` |
| 找不到模型 | 下载失败或路径错误 | 检查 `results/20260201_175440_sonic/` 确认文件存在 |
| 权重加载失败 | 单文件 vs 目录式格式不匹配 | 已通过 Step 4.5 修复 |
| 交互模式输入超范围 | 输入不在 0-15 范围内 | 重新输入有效数字 (0-15) 或 'q' 退出 |

---

## 性能指标

- **数据加载**: 41242 frames 耗时 ~2 秒
- **单帧推理速度**: 几百 ms（依赖 GPU 利用率）
- **显存占用**: ~3.4 GB / 8 GB（RTX 3060 Ti）
- **生成质量**: MSE ≈ 0.078（与真实帧对比）

---

## 总结

✅ **完整的 Windows + NVIDIA GPU 推理流程已验证通过**
- 环境配置正确
- 权重兼容性已解决
- 推理结果可视化已生成
- 交互模式可正常使用

📌 **下一步**可以尝试：
1. 改变超参数 (generation_steps, temperature 等)
2. 切换不同数据集 (需先下载)
3. 训练自己的模型 (见 scripts/train_*.py)
