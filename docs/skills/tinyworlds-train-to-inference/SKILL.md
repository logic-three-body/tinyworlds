---
name: tinyworlds-train-to-inference
description: "Run TinyWorlds ZELDA as a closed loop from training to inference validation with evidence outputs: stop stale training processes, audit 2025 ZELDA git history, run gated training in order (video-to-latent-to-dynamics), run inference gate, auto-retrain dynamics on failure, and run W&B API-first plotting analysis with local Logs fallback. Use when users ask for train-to-inference automation, checkpoint readiness validation, or fixing train-pass but inference-fail issues with retrospective evidence."
---

# TinyWorlds Train-To-Inference

## Goal
Use one deterministic loop:
1. Stop stale training processes.
2. Enforce single active training launcher (kill extra stage processes).
3. Monitor dual-GPU activity during training chunks.
4. Audit git history (default: 2025 strict ZELDA evidence).
5. Run full gated training in order (`video_tokenizer -> latent_actions -> dynamics`).
6. Run standard inference validation (`tinyworlds-standard-inference-test`).
7. Check `inference_results` + metric gate.
8. If fail, retune/retrain `dynamics` and re-run inference.
9. Run W&B stage analysis (API first, local Logs fallback).
10. Write retrospective + evidence reports under `docs/action/`.

## Config Alignment Rule (ZELDA)
Before running the loop, enforce cross-stage backbone alignment (following `d574160`'s alignment principle):
- `patch_size`, `context_length`, `frame_size`, `latent_dim`, `num_bins`, `n_actions` from `configs/training.yaml`.
- Shared backbone parameters aligned across `video/latent/dynamics`:
  - `embed_dim`
  - `num_heads`
  - `hidden_dim`
  - `num_blocks`

Do not keep stale hardcoded checkpoint paths in `configs/dynamics.yaml`; prefer CLI/runtime injection.

## Single Command (Recommended)
Run from repo root:

```bash
python docs/skills/tinyworlds-train-to-inference/scripts/train_to_inference_loop.py \
  --repo-root . \
  --python-exec ./.venv/bin/python \
  --torchrun ./.venv/bin/torchrun \
  --train-stage all \
  --checkpoint-policy fresh \
  --video-chunk 200 \
  --latent-chunk 200 \
  --dynamics-chunk 200 \
  --target-updates 200 \
  --retry-chunk 200 \
  --retry-init-learning-rate 0.0005 \
  --init-log-interval 100 \
  --cleanup-extra-processes true \
  --enforce-dual-gpu-check true \
  --monitor-interval-sec 5 \
  --gpu-util-threshold 1 \
  --gpu-required-samples 2 \
  --max-mse 0.03 \
  --max-inference-retries 2 \
  --enable-wandb-analysis true \
  --wandb-source api_first \
  --history-audit-year 2025 \
  --history-audit-mode strict
```

## Inference Pass/Fail Gate
Pass only when all conditions hold:
- `scripts/run_inference.py` exits successfully.
- Logs include:
  - selected checkpoints for all 3 models
  - `Inferring frame ...`
  - `Inference stats`
- A new PNG appears under `inference_results/inference_results_gt_vs_pred_*.png`.
- `Mean Squared Error (GT vs Pred)` is present and `<= max_mse`.

Fail when any condition above is missing.

## Retune Policy on Inference Failure
If inference fails:
1. Continue from latest valid `dynamics` checkpoint.
2. Increase dynamics target updates by one retry chunk (default `+2000`).
3. Lower dynamics init LR each retry (`retry_init_learning_rate * retry_lr_decay`).
4. Re-run inference gate.
5. Stop when pass or retry budget reached.

## Artifacts
- Runtime report:
  - `docs/action/train-to-inference-report-<timestamp>.md`
- Git evidence report:
  - `docs/action/zelda-<year>-git-evidence-<timestamp>.md`
- W&B analysis reports:
  - `docs/action/wandb-analysis-<timestamp>.md`
  - `docs/action/wandb-analysis-<timestamp>.json`
  - `docs/action/plots/wandb-analysis-<timestamp>/*.png`
- Retrospective report (required):
  - `docs/action/train-to-inference-retrospective-<timestamp>.md`
- Training reports from controller:
  - `docs/action/auto-training-loop-report-<timestamp>.md`
- Inference outputs:
  - `inference_results/*.png`

## W&B Analysis Template
Use stage plots below:
1. `video_tokenizer`: `train/loss`, `train/codebook_usage`, `learning_rate/group_0`.
2. `latent_actions`: `train/loss`, `action_entropy + unique_actions`, `latent_actions/codebook_usage`, `learning_rate/group_0`.
3. `dynamics`: `train/loss (+rolling)`, `first10/last10 loss`, `action_entropy + unique_actions`, `learning_rate/group_0`.

Data source priority:
1. W&B API (`wandb.Api`) with stage run IDs.
2. Local `Logs/*history*.csv|json` fallback if API is unavailable.

## Hard Rules
1. Default to full pipeline (`--train-stage all`) for closed-loop validation.
2. Default to `--checkpoint-policy fresh` for clean run unless user requests warm start.
3. Training is not complete until inference gate passes (`MSE <= max_mse` for ZELDA default `0.03`).
4. Closed-loop pass requires all reports: train-to-inference report, retrospective, git evidence, W&B analysis.
5. Keep exactly one active training launcher per chunk; terminate extra stage processes immediately.
6. For dual-GPU default (`nproc_per_node=2`), require evidence that both GPUs were active.
7. Keep cross-stage backbone aligned and avoid stale hardcoded checkpoint paths in configs.

## Validated Example (v2)
- Teacher-forced closed-loop pass (single GPU):
  - command core: `train_to_inference_loop.py --train-stage none --teacher-forced --max-mse 0.03`
  - report: `docs/action/train-to-inference-report-2026_02_09_19_04_19.md`
  - retrospective: `docs/action/train-to-inference-retrospective-2026_02_09_19_04_19.md`
  - result: `status=passed`, `MSE=0.012611`.
- Dual-GPU monitor and process-guard validation:
  - command core: `auto_train_loop.py --nproc-per-node 2 --cleanup-extra-processes true --enforce-dual-gpu-check true --only-stage video_tokenizer`
  - report: `docs/action/auto-training-loop-report-2026_02_09_19_10_35.md`
  - evidence: gate line contains `launcher <pid>` and `dual_gpu_ok True`.

## Known Gap
- Strict autoregressive inference (`teacher_forced=false`) is still above `0.03` in current quick-scale checkpoints.
- Practical baseline from this run:
  - tokenizer A/B recon MSE is around `0.034~0.046`.
  - autoregressive GT-vs-Pred MSE observed around `0.04~0.06`.
- Action item: when strict autoregressive gate is required, increase tokenizer quality first, then retrain latent/dynamics with aligned checkpoints.

## Resources
- Gated training controller:
  - `docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py`
- Standard inference entrypoint:
  - `scripts/run_inference.py`
- This skill automation wrapper:
  - `docs/skills/tinyworlds-train-to-inference/scripts/train_to_inference_loop.py`
- W&B stage analyzer:
  - `docs/skills/tinyworlds-train-to-inference/scripts/wandb_stage_analyzer.py`
- Git evidence audit:
  - `docs/skills/tinyworlds-train-to-inference/scripts/git_zelda_audit.py`
