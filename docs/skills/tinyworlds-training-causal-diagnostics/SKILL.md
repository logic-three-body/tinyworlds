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

## Workflow
### Step 1: Preflight
Before running:
- Read `configs/training.yaml`, `configs/video_tokenizer.yaml`, `configs/latent_actions.yaml`, `configs/dynamics.yaml`.
- Confirm dual-GPU default unless user says otherwise:
  - `distributed.use_ddp=true`
  - `nproc_per_node=2`
- Check there is no conflicting active training process.
- Define gate schedule and retry budget from `references/auto-training-gates.md`.

### Step 2: Run Stage in Chunks
Run each stage in chunked updates (gate-sized), not one long run.

Use stage scripts directly:
- `scripts/train_video_tokenizer.py`
- `scripts/train_latent_actions.py`
- `scripts/train_dynamics.py`

Pass overrides via CLI for each chunk:
- `n_updates=<chunk_updates>`
- `checkpoint=<best_checkpoint_or_null>`
- keep stage-specific config path + `--training_config configs/training.yaml`

If user asks for unattended execution, prefer running:
```bash
python docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py \
  --repo-root . \
  --nproc-per-node 2
```

For dynamics-only loop with fixed upstream checkpoints:
```bash
python docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py \
  --repo-root . \
  --nproc-per-node 2 \
  --only-stage dynamics \
  --video-checkpoint results/<run>/video_tokenizer/checkpoints \
  --latent-checkpoint results/<run>/latent_actions/checkpoints
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
- Inspect latest visualization files under `results/<run>/<stage>/visualizations/`.
- Validate checkpoint integrity (non-empty model/optimizer state).

### Step 4: Decide Continue vs Retune
Per gate, produce exactly one decision:
- `continue`: stage is healthy, run next chunk.
- `retune`: adjust config using ladder rules, restart chunk from best valid checkpoint.
- `restart_stage`: severe collapse or unusable checkpoints; restart stage from stable checkpoint or fresh.

Use mutation ladders in `references/auto-training-gates.md`.

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
- Run a short inference smoke test.
- Write a report in `docs/action/` with:
  - gate-by-gate decisions,
  - config changes applied,
  - final checkpoint paths,
  - residual risks.

## Hard Rules
1. Do not stop at first failure; apply retune ladder and retry until retry budget is exhausted.
2. Keep dual-GPU DDP by default and optimize stability before increasing model size.
3. Never treat a tiny/empty checkpoint as valid fallback.
4. If `latest` checkpoint is invalid, pin to newest valid checkpoint explicitly.

## Resources
- Gate criteria and mutation ladders: `references/auto-training-gates.md`
- Root-cause diagnostics checklist: `references/causal-diagnostic-checklist.md`
- Numeric gate scorer: `scripts/stage_gate_verdict.py`
- Full controller loop: `scripts/auto_train_loop.py`
