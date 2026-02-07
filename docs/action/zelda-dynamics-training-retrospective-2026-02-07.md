# Zelda Dynamics 训练反思（2026-02-07）

## 1. 结论摘要
- 本次 dynamics 成功不是单一因素，而是“参数修正 + 训练流程修正”的组合结果。
- 旧版本参数的核心问题是：`learning_rate=0.01` 与 `n_updates=300000` 组合，使训练前中期长期处于高学习率区间，导致 loss 持续上行。
- 新版本参数的核心改动是：`learning_rate=0.0005`、`gradient_accumulation_steps=2`、`target_updates=5000`，让优化步长和观测周期都回到可控区间。

## 2. 证据与对比

### 2.1 旧版（劣化）观测
数据源：`Logs/dynamics_history_2pghxx71_latest.csv`

- steps: `0 -> 50343`
- loss: `1.7785 -> 6.0698`（显著恶化）
- learning_rate: `max=0.01`，`last=0.009629`
- 分窗 loss 均值：
- `0-10k`: `3.8657`
- `10k-20k`: `5.0537`
- `20k-30k`: `5.7302`
- `30k-40k`: `6.0434`
- `40k-50k+`: `6.4226`

解释：在 300k 总步长的 cosine 调度下，50k 仍处于高学习率段（`~0.0096`），模型一直在“高步长震荡”，没有进入稳定收敛区。

### 2.2 新版（成功）观测
数据源：
- `Logs/dynamics_history_66wg2d11_gate_dynamics.csv`
- `Logs/auto_train_dynamics_only_fix_20260207_200259.log`

- steps: `0 -> 4999`（完成）
- loss: `3.5564 -> 2.5335`（持续下降）
- learning_rate: `max=0.0005`，`last=0.000005`
- 分窗 loss 均值：
- `0-1000`: `3.1154`
- `1000-2000`: `2.8112`
- `2000-3000`: `2.7447`
- `3000-4000`: `2.7074`
- `4000-5000`: `2.7207`

解释：新版在 5k 步内完成 warmup+衰减，优化强度从“可学习”逐步进入“可收敛”，没有出现旧版持续抬升。

## 3. 为什么旧参数会导致 dynamics 劣化

### 3.1 学习率绝对值过高
- 旧版 `learning_rate=0.01` 对当前 dynamics 任务过激。
- dynamics 是离散 token 条件预测，梯度噪声本身较高，高 lr 会放大误差更新，形成“越训越偏”。

### 3.2 学习率调度与总步长不匹配
- 旧版 `n_updates=300000` 导致早中期很长时间都停留在高 lr 区间。
- 训练在已经出现劣化信号时，lr 还没有降到足够低，无法自我纠偏。

### 3.3 梯度累积过大（放大有效更新）
- 旧版 `gradient_accumulation_steps=4`（配合双卡、每卡 batch 8）使有效 batch 与每次参数更新强度偏大。
- 在高 lr 前提下，过大的有效更新更容易跨过稳定区间，表现为 loss 窗口均值持续走高。

## 4. 为什么新参数能成功

### 4.1 初始学习率回到稳定区间
- 新版 `learning_rate=0.0005`，相比旧版下降 20x。
- 直接降低了每步参数扰动幅度，训练从“发散边缘”回到“可优化区间”。

### 4.2 梯度累积从 4 降到 2
- 新版 `gradient_accumulation_steps=2`，降低了单次更新的等效冲击。
- 对动态预测这类高噪声任务，更容易获得平滑下降曲线。

### 4.3 先用 5k 步闭环验证
- 新版使用 `target_updates=5000` 做阶段性闭环，先确认收敛形态正确，再考虑延长总步数。
- 相比一次性跑 300k，这种策略能更早发现方向错误并避免资源浪费。

## 5. 重要补充：并非只有参数问题

本轮还修复了双卡 checkpoint 写盘竞态，否则会把参数效果判断混淆。

证据：
- 旧 run checkpoint 完整性：`results/2026_02_06_00_44_31/dynamics/checkpoints` 中 `total=28, valid=3, bad=25`
- 新 run checkpoint 完整性：`results/auto_train_loop_2026_02_07_20_02_59/dynamics/checkpoints` 中 `total=3, valid=3, bad=0`

代码修复点：`scripts/train_dynamics.py`
- 增加从 checkpoint 恢复 `start_step`、optimizer、scheduler
- 分布式下仅主进程写 checkpoint，并在写盘前后 barrier 同步

结论：旧版“劣化”是参数不稳与写盘机制缺陷叠加；新版成功是两者同时修正后的结果。

## 6. 后续建议（可执行）
- dynamics 默认初始参数建议：
- `learning_rate=5e-4`
- `gradient_accumulation_steps=2`
- `batch_size_per_gpu=8`（双卡）
- 采用分段训练：每 `5k` 为一个门控段，达标再继续。
- 持续保留“仅 rank0 写 checkpoint + resume optimizer/scheduler”的训练脚本约束，避免再次出现“看似继续训练，实际从坏点重启”的假象。
