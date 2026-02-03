## 最短验证 Plan：Clamp detokenize 输出，观察 rollout 是否更稳定（Windows + SONIC）

**目标**：验证"推理很快花屏"是否主要由 **detokenize 输出超出 [-1,1]** 导致分布漂移。  
做法：在每一步生成的新帧（detokenize 后）立即 `clamp(-1,1)`，再进入下一步 tokenize/dynamics。

---

### Step 1：修改 `scripts/run_inference.py`（最小改动）

**定位**：在生成循环里，找到从 dynamics 输出 latents/tokens 并执行：

```python
pred_frame = video_tokenizer.detokenize(pred_latents_or_tokens)
```

**插入 clamp（必须）**：在 `generated_frames.append(...)` 之前增加一行：

```python
pred_frame = pred_frame.clamp(-1.0, 1.0)
```

> 如果代码里变量名不是 `pred_frame`，按实际变量名改；原则是：**detokenize → clamp → append/下一步 tokenize**。

---

### Step 2（可选，但推荐）：打印范围验证 clamp 生效

在 clamp 前后各打印一次（只打印前 1~2 步即可，避免刷屏）：

```python
if step_idx < 2:
    print("pre-clamp min/max:", pred_frame.min().item(), pred_frame.max().item())
pred_frame = pred_frame.clamp(-1.0, 1.0)
if step_idx < 2:
    print("post-clamp min/max:", pred_frame.min().item(), pred_frame.max().item())
```

---

### Step 3：运行推理（保持其它参数不变）

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
python scripts/run_inference.py --config configs/inference.yaml
```

---

### Step 4：判定结果（验收标准）

✅ **Clamp 生效**：日志里 post-clamp 的 min/max 应该落在 `[-1,1]`
✅ **Rollout 更稳定（期望）**：`GT vs Pred` 图里从 Pred3 开始的崩坏速度变慢、结构保持更久
✅ 输出仍生成：

* `inference_results_gt_vs_pred_*.png`
* `inference_video_*.mp4`

---

### 结论解读

* 如果 clamp 后画面明显更稳：说明之前主要是 **输出域漂移** 导致 tokenizer/dynamics 输入跑飞
* 如果 clamp 后改善不大：主要问题更可能是 **随机 action + rollout 误差累积**（下一步再改 action 策略）
