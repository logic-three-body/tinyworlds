# Zelda 训练因果诊断报告（2026-02-07）

**生成时间**: 2026-02-07 17:10 CST  
**结论级别**: `critical`（主因在 dynamics 优化配置与 checkpoint 写入路径，不是 latent_actions 单点失效）

---

## 1. 证据来源

- 指标导出:
  - `Logs/video_tokenizer_history_okmjny24_latest.csv`
  - `Logs/latent_actions_history_d376skck_latest.csv`
  - `Logs/dynamics_history_2pghxx71_latest.csv`
  - `Logs/dynamics_history_2pghxx71_refresh.csv`（本次刷新，step 到 `54112`）
- 可视化:
  - `results/2026_02_05_21_05_51/latent_actions/visualizations/`
  - `results/2026_02_06_00_44_31/dynamics/visualizations/`
- checkpoint:
  - `results/2026_02_05_21_05_51/latent_actions/checkpoints/`
  - `results/2026_02_06_00_44_31/dynamics/checkpoints/`
- 配置与训练逻辑:
  - `configs/training.yaml`
  - `configs/latent_actions.yaml`
  - `configs/dynamics.yaml`
  - `scripts/train_dynamics.py`
  - `utils/utils.py`
  - `utils/scheduler_utils.py`

---

## 2. 综合判断

## 2.1 latent_actions 是否导致 dynamics 劣化

**判断**: `latent_actions` 是次要风险，不是 dynamics 持续劣化的主因。

证据:
- latent 收敛正常:
  - loss `1.0465 -> 0.02886`（`Logs/latent_actions_history_d376skck_latest.csv`）
- latent 多样性确有回落，但未塌缩:
  - early/mid/late `action_entropy` 均值: `0.927 / 1.023 / 0.744`
  - early/mid/late `unique_actions` 均值: `3.50 / 4.14 / 2.57`
  - early/mid/late `codebook_usage` 均值: `0.219 / 0.259 / 0.161`
  - 说明后段动作分布变窄，但并非“单动作锁死”
- dynamics 内部动作统计基本稳定:
  - early/mid/late `action_entropy` 均值: `0.7689 / 0.7683 / 0.7684`
  - early/mid/late `unique_actions` 均值: `2.662 / 2.661 / 2.661`
- 相关性很弱:
  - `corr(loss, action_entropy) = -0.042`
  - `corr(loss, unique_actions) = -0.045`

结论:
- latent 的“动作多样性偏低”会压上限，但不足以解释 dynamics loss 的持续上扬。

## 2.2 dynamics 本身导致的劣化

**判断**: dynamics 自身优化配置与保存链路问题是主因。

证据:
- loss 持续恶化:
  - `1.778 -> 7.426`（`step 0 -> 54112`）
  - early/mid/late loss 均值: `4.547 / 5.512 / 6.441`
  - first10% vs last10% 均值差: `+4.509`
- LR 处在长期高位:
  - `configs/dynamics.yaml` 中 `learning_rate: 0.01`, `n_updates: 300000`
  - `step 54112` 时学习率仍约 `0.00955`（`Logs/dynamics_history_2pghxx71_refresh.csv`）
  - `corr(loss, lr) = 0.449`（中等偏强正相关）
- checkpoint 可恢复性差:
  - dynamics 共 `28` 个 checkpoint，仅 `3` 个有效（`26000/44000/52000`）
  - 最新 `dynamics_step_54000` 无效（900B 空 dict）
  - 最新可恢复点: `results/2026_02_06_00_44_31/dynamics/checkpoints/dynamics_step_52000`

结论:
- 当前 dynamics 已出现“训练继续但质量劣化 + 部分 checkpoint 不可用”的典型异常轨迹。

---

## 3. 配置修改建议（保留双卡，提升有效利用）

> 目标不是盲目拉高 GPU 占用率，而是提高“有效吞吐 + 可恢复性 + 收敛稳定性”。当前 GPU 实测占用已高（约 `97-99%`）。

## 3.1 立即生效（当前 dynamics run 续训建议）

`configs/training.yaml`:

```yaml
distributed:
  use_ddp: true
nproc_per_node: 2
compile: true
preload_ratio: 0.75
```

`configs/dynamics.yaml`:

```yaml
batch_size_per_gpu: 8
gradient_accumulation_steps: 2
n_updates: 120000
learning_rate: 0.0005
log_interval: 1000
checkpoint: /root/tinyworlds/results/2026_02_06_00_44_31/dynamics/checkpoints/dynamics_step_52000
```

说明:
- `lr 0.01 -> 5e-4` 是核心止损项。
- `grad_accum 4 -> 2` 缩短一次优化反馈链路，便于更快修正发散趋势。
- 保留双卡 DDP，不退化到单卡。

## 3.2 下一轮联动重训（latent + dynamics）

`configs/latent_actions.yaml`:

```yaml
n_updates: 15000
learning_rate: 0.00008
log_interval: 250
```

`configs/training.yaml`（下一轮可选）:

```yaml
n_actions: 8
```

说明:
- `n_actions: 8` 有利于提升动作桶利用率，但会影响联动训练，需 latent+dynamics 一起重训。

---

## 4. 非配置项（必须同步处理）

该项不是 YAML 能解决的问题，但对双卡稳定运行是必须项:

- 训练脚本中 checkpoint 保存目前是多 rank 同路径写入（`scripts/train_dynamics.py` + `utils/utils.py`），会出现非主进程空状态覆盖。
- 需要改成仅 `is_main` 写 checkpoint，或按 rank 区分输出目录后再聚合。

---

## 5. 优先动作

1. 立即以 `dynamics_step_52000` 作为恢复点，按 3.1 配置续训。  
2. 先跑 `5k` 更新量做 smoke，检查 loss 窗口均值是否回落。  
3. 同时修复“多 rank checkpoint 覆盖”后再进入长跑。  
4. 下一轮再做 3.2 的 latent+dynamics 联动重训。  
