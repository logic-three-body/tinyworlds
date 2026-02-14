---
name: tinyworlds-training-causal-diagnostics
description: "Run TinyWorlds as an automated closed-loop training controller: start training, pause at gate steps, evaluate Logs and visualizations, continue if healthy, otherwise patch configs and retrain from best checkpoint until all stages pass quality gates. Use when users ask for iterative/automatic training with step-based diagnostics and self-correcting retries."
---

# TinyWorlds Auto Training Loop

## Overview
Use this skill to execute TinyWorlds training end-to-end with automatic gates:
1. Start stage training.
2. Stop at configured gate steps.
3. Export metrics and inspect visualizations.
4. Choose `continue` or `retune+restart`.
5. Repeat until stage passes, then move to next stage.

The loop target is completion of all stages: `video_tokenizer -> latent_actions -> dynamics`.

## Output Governance
Use `docs/data-output-governance.md` as the source of truth.

Define:
- `RUN_BUNDLE_ROOT=/mnt/y/WorldModel/Tinyworld_backup/<YYYYMMDD_HHMMSS>`

All generated artifacts must be written under:
- `RUN_BUNDLE_ROOT/results/...`
- `RUN_BUNDLE_ROOT/inference_results/...`
- `RUN_BUNDLE_ROOT/wandb/run-*/...`
- `RUN_BUNDLE_ROOT/docs/action/...` (generated reports/plots)

Temporary migration logs are local-only and must remain git-ignored:
- `docs/action/asset-move-report-*.md`
- `docs/action/asset-move-manifest-*.txt`

## Workflow
### Step 1: Preflight
Before running:
- Read `configs/training.yaml`, `configs/video_tokenizer.yaml`, `configs/latent_actions.yaml`, `configs/dynamics.yaml`.
- Confirm dual-GPU default unless user says otherwise:
  - `distributed.use_ddp=true`
  - `nproc_per_node=2`
- Enforce single active launcher: terminate extra stage processes before each chunk.
- Enable dual-GPU runtime evidence checks while each chunk is running.
- Define gate schedule and retry budget from `references/auto-training-gates.md`.

### Step 2: Run Stage in Chunks
Run each stage in chunked updates (gate-sized), not one long run.

Use stage scripts directly:
- `scripts/train_video_tokenizer.py`
- `scripts/train_latent_actions.py`
- `scripts/train_dynamics.py`

Pass overrides via CLI for each chunk:
- `n_updates=<absolute_target_updates_for_this_chunk>` (absolute-step resume semantics)
- `checkpoint=<best_checkpoint_or_null>`
- keep stage-specific config path + `--training_config configs/training.yaml`

Resume support by stage:
- `--only-stage video_tokenizer` accepts `--video-checkpoint`.
- `--only-stage latent_actions` accepts `--latent-checkpoint`.
- `--only-stage dynamics` requires both `--video-checkpoint` and `--latent-checkpoint` (optional `--dynamics-checkpoint`).

### Stage Parameter Profiles
Config baseline (`configs/*.yaml`):
| Stage | batch_size_per_gpu | grad_accum | learning_rate | log_interval | n_updates |
|---|---:|---:|---:|---:|---:|
| video_tokenizer | 16 | 2 | 4e-4 | 2500 | 50000 |
| latent_actions | 16 | 1 | 1e-4 | 500 | 15000 |
| dynamics | 12 | 1 | 2e-4 | 1000 | 160000 |

Long-run safe start (recommended):
| Stage | batch_size_per_gpu | grad_accum | learning_rate | chunk | target_updates |
|---|---:|---:|---:|---:|---:|
| video_tokenizer | 8 | 2 | 3e-4 | 5000 | 40000 |
| latent_actions | 8 | 1 | 1e-4 | 2000 | 10000 |
| dynamics | 8 | 1 | 2e-4 | 4000 | 160000 |

Track and report:
- `effective_batch = batch_size_per_gpu * gradient_accumulation_steps * nproc_per_node`

If user asks for unattended execution, prefer running:
```bash
python docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py \
  --repo-root . \
  --nproc-per-node 2 \
  --cleanup-extra-processes true \
  --enforce-dual-gpu-check true \
  --video-min-batch-size 8 \
  --latent-min-batch-size 8 \
  --dynamics-min-batch-size 4
```

For dynamics-only loop with fixed upstream checkpoints:
```bash
python docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py \
  --repo-root . \
  --nproc-per-node 2 \
  --cleanup-extra-processes true \
  --enforce-dual-gpu-check true \
  --only-stage dynamics \
  --video-checkpoint <RUN_BUNDLE_ROOT>/results/<run>/video_tokenizer/checkpoints \
  --latent-checkpoint <RUN_BUNDLE_ROOT>/results/<run>/latent_actions/checkpoints \
  --dynamics-min-batch-size 4
```

For long continuation from an existing dynamics checkpoint (example to reach +124k updates):
```bash
python docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py \
  --repo-root . \
  --nproc-per-node 2 \
  --cleanup-extra-processes true \
  --enforce-dual-gpu-check true \
  --only-stage dynamics \
  --video-checkpoint <RUN_BUNDLE_ROOT>/results/<run>/video_tokenizer/checkpoints \
  --latent-checkpoint <RUN_BUNDLE_ROOT>/results/<run>/latent_actions/checkpoints \
  --dynamics-checkpoint <RUN_BUNDLE_ROOT>/results/<run>/dynamics/checkpoints \
  --target-updates 124000 \
  --dynamics-chunk 4000 \
  --init-learning-rate 0.0005 \
  --init-grad-accum 2 \
  --init-log-interval 1000 \
  --dynamics-min-batch-size 4
```

### Step 3: Export and Evaluate at Each Gate
After each chunk:
- Export run history to `Logs/` (use existing exporter):
```bash
python docs/skills/tinyworlds-training-diagnostics/scripts/export_local_wandb_history.py \
  --run "<stage>=wandb/run-<run_ts>-<run_id>/run-<run_id>.wandb" \
  --out-dir Logs \
  --suffix gate
```
- Run `scripts/stage_gate_verdict.py` for numeric verdict.
- Inspect latest visualization files under `<RUN_BUNDLE_ROOT>/results/<run>/<stage>/visualizations/`.
- Validate checkpoint integrity (non-empty model/optimizer state).
- Enforce visualization freshness: if no new `.png` appears for a chunk, treat as gate failure and retune/retry.

### Step 4: Decide Continue vs Retune
Per gate, produce exactly one decision:
- `continue`: stage is healthy, run next chunk.
- `retune`: adjust config using ladder rules, restart chunk from best valid checkpoint.
- `restart_stage`: severe collapse or unusable checkpoints; restart stage from stable checkpoint or fresh.

Use mutation ladders in `references/auto-training-gates.md`.
Runtime retune ladder (implemented by controller):
- Runtime error/OOM: reduce `batch_size_per_gpu` first (`/2`) with stage floors:
  - `video_tokenizer` floor `8`
  - `latent_actions` floor `8`
  - `dynamics` floor `4`
- `video_tokenizer` gate fail: `learning_rate * 0.5`, `gradient_accumulation_steps + 1`, `log_interval` floor `500`.
- `latent_actions` gate fail: `learning_rate * 0.8`, `log_interval` floor `250`.
- `dynamics` gate fail: `learning_rate * 0.2`, `gradient_accumulation_steps - 1`, `log_interval` floor `500`.

### Step 5: Convergence and Handoff
A stage is complete only when:
- Final target steps are reached.
- Gate verdict remains healthy in last gate window.
- Latest checkpoint is resumable.

Then hand off checkpoint paths:
- `video_tokenizer_path` for latent/dynamics.
- `latent_actions_path` for dynamics.

### Step 6: Finish Full Pipeline
After `dynamics` passes:
- Run `docs/skills/tinyworlds-standard-inference-test` immediately (no manual gap).
- Check `<RUN_BUNDLE_ROOT>/inference_results/inference_results_gt_vs_pred_*.png` and parsed MSE.
- For ZELDA default gate: `Mean Squared Error (GT vs Pred) <= 0.03`.
- If inference gate fails, return to `dynamics` retune/retry loop, then re-run inference gate.
- Write a report in `<RUN_BUNDLE_ROOT>/docs/action/` with:
  - gate-by-gate decisions,
  - config changes applied,
  - final checkpoint paths,
  - residual risks.

## Hard Rules
1. Do not stop at first failure; apply retune ladder and retry until retry budget is exhausted.
2. Keep dual-GPU DDP by default and optimize stability before increasing model size.
3. Never treat a tiny/empty checkpoint as valid fallback.
4. If `latest` checkpoint is invalid, pin to newest valid checkpoint explicitly.
5. Keep exactly one active training launcher; kill extras and keep the current run process only.
6. For `nproc_per_node >= 2`, require dual-GPU runtime evidence, otherwise fail gate and retry.
7. Training is not considered complete until standard inference gate passes.
8. If a stage reaches its minimum batch floor and still fails, mark run as degraded and log reason explicitly.
9. Do not interpret chunk-local progress as global completion; completion follows absolute target step.
10. Output data paths must follow `docs/data-output-governance.md`; `asset-move-report/manifest` files are local-only and must not be committed.

## Validated Example
- Dual-GPU launcher/monitor success sample:
  - command core: `auto_train_loop.py --nproc-per-node 2 --only-stage video_tokenizer --target-updates 200 --video-chunk 200 --cleanup-extra-processes true --enforce-dual-gpu-check true --video-min-batch-size 8`
  - report: `docs/action/auto-training-loop-report-2026_02_09_19_10_35.md`
  - gate evidence: `decision watch`, `launcher <pid>`, `dual_gpu_ok True`.

## Resources
- Gate criteria and mutation ladders: `references/auto-training-gates.md`
- Root-cause diagnostics checklist: `references/causal-diagnostic-checklist.md`
- Numeric gate scorer: `scripts/stage_gate_verdict.py`
- Full controller loop: `scripts/auto_train_loop.py`
