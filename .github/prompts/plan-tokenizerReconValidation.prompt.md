````markdown
## 任务：在 tinyworlds 推理入口加入 Tokenizer Recon 验证（修正版 Plan）

**目标**：验证 `video_tokenizer` 是否能把输入帧 **encode(tokenize) → decode(detokenize)** 还原回接近原图，输出 **Recon MSE** 和 **GT vs Recon 对比图**，用于判断“花屏”是否来自 tokenizer 还是后续 dynamics rollout / action。

**修改范围**：只改 `scripts/run_inference.py`（尽量复用现有可视化风格；不改主流程结构）。

---

## 需求细节

### 1) 插入位置（必须）
在 `run_inference.py` 中，读取完 `ground_truth_frames` 并切出：

```python
context_frames = ground_truth_frames[:, :args.context_window, :, :, :]
generated_frames = context_frames.clone()
````

之后、**进入生成循环之前**，插入 recon 验证代码。

---

### 2) Recon 流程（必须）

使用与推理一致的 tokenizer 编解码路径，且不产生梯度：

```python
with torch.inference_mode():
    # 输入：context_frames [B, T, C, H, W]
    idx = video_tokenizer.tokenize(context_frames)
    lat = video_tokenizer.quantizer.get_latents_from_indices(idx)
    recon = video_tokenizer.detokenize(lat)
    # recon shape == context_frames shape
```

> 说明：确保 `context_frames` 与 `video_tokenizer` 在同一 device（跟随 `args.device`），并 `video_tokenizer.eval()`。

---

### 3) 计算并打印（必须）

用 float 计算 MSE，并打印到控制台：

```python
mse = torch.mean((recon.float() - context_frames.float()) ** 2).item()
print(f"Tokenizer recon MSE: {mse:.6f}")
```

---

### 4) 可视化保存（必须，且不要写死归一化区间）

* 输出：GT vs Recon 对比 PNG
* 目录：`inference_results/`
* 文件名：`tokenizer_recon_gt_vs_recon_<timestamp>.png`
* 布局：2 行 x T 列（T = `context_window`）

  * 上排：GT
  * 下排：Recon
* **关键修正**：保存前将帧转换到 `[0,1]` 时**不要假设一定是 [-1,1]**，应做“自适应范围判断”：

  * 若 `min < 0`，按 `[-1,1] → [0,1]`：`(x + 1) / 2`
  * 若 `max > 1`，按 `[0,255] → [0,1]`：`x / 255`
  * 否则默认已是 `[0,1]`
  * 最后 `clamp(0,1)`
* PNG 保存前转回 CPU（`detach().cpu()`）

---

### 5) 不影响后续推理（必须）

* recon 验证只读 `context_frames`，不修改 `generated_frames`
* 发生异常时（例如接口差异）要给出清晰报错并不中断后续推理（可选：try/except 后继续主流程）

---

## 实现方案

### 位置 A：添加辅助函数（建议）

在 `def main():` 之前加入两个辅助函数：

1. `_to_vis(frames)`：将 `[B,T,C,H,W]` 转成 `[0,1]` 的 float CPU tensor，包含上述“自适应范围判断”
2. `_save_tokenizer_recon_visualization(gt_frames, recon_frames, mse)`：

   * 调用 `_to_vis`
   * `matplotlib` 生成 2 x T 网格图并保存
   * `plt.tight_layout()` + suptitle 显示 MSE

---

### 位置 B：插入 recon 验证代码（必须）

在 `main()` 中紧跟 `context_frames` 赋值之后插入：

* 打印头尾标识
* `torch.inference_mode()` 下跑 recon
* 打印 `Tokenizer recon MSE`
* 保存对比 PNG

---

## 验收标准

✅ 运行：

```powershell
python scripts/run_inference.py --config configs/inference.yaml
```

时，日志包含：

```
=== Tokenizer Reconstruction Sanity Check ===
Tokenizer recon MSE: <value>
Tokenizer recon visualization saved to: inference_results/tokenizer_recon_gt_vs_recon_<timestamp>.png
=== Sanity Check Complete ===
```

✅ `inference_results/` 下生成对比图 PNG（2 行 x T 列，上 GT 下 Recon）

✅ 原有推理输出（GT vs Pred PNG、MP4）仍正常生成（recon 检查不改变后续流程）

---

## 解释提示（给自己用）

* Recon 图像“像不像”比 MSE 数值区间更重要
* 若 Recon 视觉接近 GT，则 tokenizer 基本正常；后续花屏更可能来自 dynamics rollout + 随机/不匹配 action

```
::contentReference[oaicite:0]{index=0}
```
