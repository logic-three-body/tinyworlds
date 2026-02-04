# Zelda 训练对话 Checkpoint

**创建时间**: 2026-02-04 18:50 UTC+8  
**对话状态保存**: Session Checkpoint for Multi-Machine Continuation

---

## 1. 训练进度概览

### 当前状态
- **项目**: tinyworlds (Zelda 世界模型)
- **训练模式**: Single GPU Training (GPU 0)
- **当前阶段**: Video Tokenizer
- **迭代进度**: 1,009 / 40,000 (2.5%)
- **训练速度**: 2.33 iterations/sec
- **预计阶段完成**: ~4.8 小时

### 三阶段训练计划
1. ✅ **Video Tokenizer**: 2.5% progress (ETA: 4.8h)
2. ⏳ **Latent Actions**: Waiting (ETA: 2.4h)
3. ⏳ **Dynamics Model**: Waiting (ETA: 36h)
4. **总计 ETA**: 43+ 小时

### W&B 监控链接
- **Dashboard**: https://wandb.ai/1552451580-sichuan-university/tinyworlds/runs/a3el1fs3
- **Project**: tinyworlds
- **Entity**: 1552451580-sichuan-university
- **Run ID**: a3el1fs3

---

## 2. W&B API 凭证

### 登录信息
```
用户名/Entity: 1552451580-sichuan-university
项目名: tinyworlds
API Key 位置: C:\Users\PC\_netrc
```

### W&B 认证方式
在新机器上配置 W&B 时，需要在 `_netrc` 或 `~/.netrc` 中添加：
```
machine api.wandb.ai
login <your-username>
password <your-api-key>
```

**或使用命令**:
```bash
wandb login <YOUR_API_KEY>
```

---

## 3. 硬件配置

### GPU 信息
| 参数 | GPU 0 | GPU 1 |
|---|---|---|
| 型号 | RTX 4090 | RTX 4090 |
| 总显存 | 24.564 GB | 24.564 GB |
| 已用显存 | 17.8 GB (72%) | 10.0 GB (41%) |
| 计算利用率 | 88% ✅ | 12% ❌ |
| 功率 | 267W | 11W |
| 温度 | 56°C | 31°C |
| 状态 | ACTIVE_TRAINING | IDLE |

### 软件环境
- **Python**: 3.10+ (.venv)
- **PyTorch**: 2.8.0+cu126
- **CUDA**: 12.6
- **CUDA Version**: 12.9 (Driver)
- **项目路径**: d:\WorldModel\tinyworlds

---

## 4. 数据集信息

### Zelda 数据集
```yaml
名称: ZELDA
文件路径: data/zelda_frames.h5
文件大小: 1,780.38 MB
帧数: 72,410
分辨率: 128×128
帧率: 15 FPS
预加载比例: 1.0 (100% 全量加载)
```

---

## 5. 训练配置 (YAML)

### configs/training.yaml
```yaml
# wandb
use_wandb: true
wandb_project: tinyworlds

# dataset
dataset: ZELDA
preload_ratio: 1.0

# shared model params
patch_size: 4
context_length: 4
frame_size: 128
latent_dim: 5
num_bins: 4
n_actions: 16

# performance
amp: true
tf32: true
compile: false

# distributed launch
distributed: 
  use_ddp: False      # 当前单GPU，改双GPU需为 True
  use_fsdp: False
  reshard_after_forward: False
nproc_per_node: 1     # 当前是1，改双GPU需为 2
standalone: true

# stage configs
video_tokenizer_config: configs/video_tokenizer.yaml
latent_actions_config: configs/latent_actions.yaml
dynamics_config: configs/dynamics.yaml

# which stages to run
run_video_tokenizer: true
run_latent_actions: true
run_dynamics: true
```

### configs/video_tokenizer.yaml
```yaml
batch_size_per_gpu: 8
gradient_accumulation_steps: 2
n_updates: 40000
learning_rate: 0.001
log_interval: 2500

embed_dim: 32
num_heads: 8
hidden_dim: 128
num_blocks: 4

checkpoint: null
```

### configs/latent_actions.yaml
```yaml
batch_size_per_gpu: 8
gradient_accumulation_steps: 1
n_updates: 10000
learning_rate: 0.0001
log_interval: 500

embed_dim: 32
num_heads: 8
hidden_dim: 128
num_blocks: 2

checkpoint: null
```

### configs/dynamics.yaml
```yaml
batch_size_per_gpu: 8
gradient_accumulation_steps: 2
n_updates: 300000
learning_rate: 0.01
log_interval: 2000

use_actions: true

embed_dim: 32
num_heads: 8
hidden_dim: 128
num_blocks: 8

video_tokenizer_path: (auto-detected)
latent_actions_path: (auto-detected)
checkpoint: null
```

---

## 6. 结果目录结构

```
results/2026_02_04_18_39_02/
├── video_tokenizer/
│   ├── checkpoints/
│   │   └── zelda/
│   │       └── *.pth (checkpoints 保存在这里)
│   └── visualizations/
├── actions/
│   ├── checkpoints/
│   │   └── zelda/
│   └── visualizations/
└── dynamics/
    ├── checkpoints/
    │   └── zelda/
    └── visualizations/
```

**关键路径**:
- Video Tokenizer: `results/2026_02_04_18_39_02/video_tokenizer/checkpoints/zelda/`
- Latent Actions: 待生成
- Dynamics: 待生成

---

## 7. 重要发现与问题记录

### GPU 利用率
- **问题**: GPU 1 未被利用 (11% utilization)
- **原因**: 配置为单 GPU 模式 (`nproc_per_node: 1`, `use_ddp: False`)
- **背景**: 早期尝试双 GPU DDP 时遇到 PyTorch Windows 上的 `libuv` 错误，因此降级为单 GPU 稳定训练
- **改进方案**: 可在后续阶段通过以下环境变量启用双 GPU：
  ```powershell
  $env:USE_LIBUV = "0"
  $env:MPI_IMPLEMENTATION = "MPICH"
  ```

### 编译优化
- **torch.compile**: 已禁用 (`compile: false`)
- **原因**: 在 Zelda 128×128 分辨率上触发 FX node 编译错误
- **状态**: 当前以非编译模式运行，稳定可靠

---

## 8. 在新机器上继续训练的步骤

### 前置条件
1. 克隆仓库: `git clone https://github.com/AlmondGod/tinyworlds.git`
2. 创建虚拟环境: `python -m venv .venv`
3. 激活: `.\.venv\Scripts\Activate.ps1`
4. 安装 PyTorch (GPU 版本):
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
   ```
5. 安装其他依赖:
   ```bash
   pip install -r requirements-notorch.txt
   ```
6. 配置 W&B:
   ```bash
   wandb login <YOUR_API_KEY>
   ```

### 检查点恢复
如果需要恢复训练（而非重新开始）:

1. **下载已有的 checkpoints** 从原机器传输:
   ```
   results/2026_02_04_18_39_02/video_tokenizer/checkpoints/zelda/
   ```

2. **在 dynamics.yaml 中指定路径** (if resuming dynamics):
   ```yaml
   video_tokenizer_path: results/2026_02_04_18_39_02/video_tokenizer/checkpoints/zelda/<latest.pth>
   latent_actions_path: results/2026_02_04_18_39_02/actions/checkpoints/zelda/<latest.pth>
   ```

### 继续训练
```powershell
cd d:\WorldModel\tinyworlds
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"

# 单 GPU 模式（稳定）
python scripts/full_train.py --config configs/training.yaml

# 或双 GPU 模式（更快，需处理 libuv）
$env:USE_LIBUV = "0"
python -m torch.distributed.launch --nproc_per_node=2 scripts/full_train.py --config configs/training.yaml
```

---

## 9. 监控与验证

### 实时监控
- **GPU**: `nvidia-smi -l 1` (每秒刷新)
- **W&B**: 访问上述 Dashboard 链接查看 loss 曲线及指标
- **Terminal**: 查看 stdout 中的 iteration/loss 输出

### 旅程里程碑
- [ ] Video Tokenizer 完成 (ETA: 2026-02-05 ~23:00)
- [ ] Latent Actions 完成 (ETA: 2026-02-06 ~02:00)
- [ ] Dynamics 完成 (ETA: 2026-02-07 ~14:00)
- [ ] 所有 checkpoints 就位 & 推理验证

---

## 10. 附录：快速参考

### 关键命令
```powershell
# 激活环境
.\.venv\Scripts\Activate.ps1

# 设置 Python 路径
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"

# 启动训练
python scripts/full_train.py --config configs/training.yaml

# GPU 监控
nvidia-smi -l 1

# W&B 信息查询
wandb login  # shows current login
wandb status
```

### 故障排查
- **找不到数据**: 确保 `data/zelda_frames.h5` 存在
- **CUDA OOM**: 减少 `batch_size_per_gpu` (当前: 8)
- **显存溢出**: 禁用 AMP 或降低 `gradient_accumulation_steps`
- **W&B 连接失败**: 检查 API key 或运行 `wandb offline`

---

**文档更新**: 2026-02-04 18:50 UTC+8  
**下一次检查**: 预计 2026-02-05 10:00 (Video Tokenizer 进度 25%+)
