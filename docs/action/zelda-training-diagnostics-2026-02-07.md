# Zelda 训练诊断报告（2026-02-07）

**生成时间**: 2026-02-07 15:02 CST  
**结论级别**: `critical`（流程在跑，但 dynamics 已明显偏离正常收敛）

---

## 1. 数据来源

### 1.1 本次导出的最新 Logs
- `Logs/video_tokenizer_history_okmjny24_latest.json`
- `Logs/video_tokenizer_history_okmjny24_latest.csv`
- `Logs/latent_actions_history_d376skck_latest.json`
- `Logs/latent_actions_history_d376skck_latest.csv`
- `Logs/dynamics_history_2pghxx71_latest.json`
- `Logs/dynamics_history_2pghxx71_latest.csv`

### 1.2 运行与可视化证据
- `results/2026_02_04_19_51_20/video_tokenizer/visualizations/`
- `results/2026_02_05_21_05_51/latent_actions/visualizations/`
- `results/2026_02_06_00_44_31/dynamics/visualizations/`

### 1.3 checkpoint 完整性核验目录
- `results/2026_02_06_00_44_31/dynamics/checkpoints/`

---

## 2. 阶段性数值诊断

## 2.1 Video Tokenizer（`okmjny24`）
- 样本行数: `40000`
- 最大 step: `39999`
- loss: `0.1469 -> 0.001531`
- 最小/中位: `0.000432 / 0.001753`

判断：整体收敛正常。

## 2.2 Latent Actions（`d376skck`）
- 样本行数: `10000`
- 最大 step: `9999`
- loss: `1.0465 -> 0.02886`
- 最小/中位: `0.01307 / 0.02555`
- action entropy: `0.3768 ~ 1.2432`（末值 `0.4826`）
- unique actions: `2 ~ 5`（末值 `2`）

判断：可训练、可收敛，但动作多样性偏低（未崩溃）。

## 2.3 Dynamics（`2pghxx71`）
- 样本行数: `50344`
- 最大 step: `50343`
- loss: `1.7785 -> 6.0698`
- 最小/中位: `1.4388 / 5.6808`
- 速度（按 `_runtime`）:
  - overall: `0.3701 step/s`
  - recent(近5k): `0.2601 step/s`
- 剩余 step: `249657`
- ETA:
  - overall 估计: `~187.4h`
  - recent 估计: `~266.6h`

loss 分窗均值（越往后越高）:
- `0-10k`: `3.866`
- `10k-20k`: `5.054`
- `20k-30k`: `5.730`
- `30k-40k`: `6.043`
- `40k-50343`: `6.423`

判断：dynamics 训练稳定性与收敛质量异常，呈持续劣化趋势。

---

## 3. 可视化诊断

## 3.1 Video Tokenizer
观察文件（示例）:
- `results/2026_02_04_19_51_20/video_tokenizer/visualizations/video_tokenizer_recon_step_10000.png`
- `results/2026_02_04_19_51_20/video_tokenizer/visualizations/video_tokenizer_recon_step_30000.png`
- `results/2026_02_04_19_51_20/video_tokenizer/visualizations/video_tokenizer_recon_step_37500.png`

现象：重建图像在结构上可对齐，细节略有平滑，属于可接受范围。

## 3.2 Latent Actions
观察文件（示例）:
- `results/2026_02_05_21_05_51/latent_actions/visualizations/reconstructions_latent_actions_step_0.png`
- `results/2026_02_05_21_05_51/latent_actions/visualizations/reconstructions_latent_actions_step_9500.png`

现象：后期重建偏平滑，但仍有场景结构，未出现完全塌缩。

## 3.3 Dynamics
观察文件（示例）:
- `results/2026_02_06_00_44_31/dynamics/visualizations/dynamics_prediction_step_10000.png`
- `results/2026_02_06_00_44_31/dynamics/visualizations/dynamics_prediction_step_26000.png`
- `results/2026_02_06_00_44_31/dynamics/visualizations/dynamics_prediction_step_34000.png`
- `results/2026_02_06_00_44_31/dynamics/visualizations/dynamics_prediction_step_44000.png`
- `results/2026_02_06_00_44_31/dynamics/visualizations/dynamics_prediction_step_50000.png`

现象：大量纯色块、条纹化、低信息输出，且在后段持续存在。与 README 中 dynamics 目标（学习可用下一帧动力学）不一致。

---

## 4. Checkpoint 完整性（高优先级异常）

核验 `results/2026_02_06_00_44_31/dynamics/checkpoints/dynamics_step_*`:
- 总 checkpoint: `26`
- 可用 checkpoint: `2`（step `26000`, `44000`）
- 异常 checkpoint: `24`（多数 `model_state_dict.pt/optim_state_dict.pt` 为极小占位，加载后为空 dict）

结论：`dynamics` 最新 checkpoint（含 `50000`）默认不可信，恢复应优先使用 `step_44000`。

---

## 5. 整体判断

- 训练流程状态：**未中断**（进程在跑，GPU 利用高）
- 训练质量状态：
  - Video Tokenizer: 正常
  - Latent Actions: 基本正常（多样性偏低）
  - Dynamics: 明显异常（数值劣化 + 可视化塌缩 + checkpoint 完整性问题）

**最终结论**：当前整体流程已偏离正常训练轨道，关键问题集中在 `dynamics`。

---

## 6. 建议动作（按优先级）

1. 先冻结当前 run，保留可用恢复点 `dynamics_step_44000`。  
2. 优先修复 checkpoint 保存策略（仅主进程写入，避免空权重覆盖）。  
3. 修复后再继续 dynamics 长跑，并在 `10k` 步窗口复查 loss 窗口均值与可视化。  
4. 在继续训练前，做一次短程 smoke（1k~2k step）验证 checkpoint 可恢复性。  

