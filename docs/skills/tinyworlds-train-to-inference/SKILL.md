---
name: tinyworlds-train-to-inference
description: "Run TinyWorlds ZELDA as a closed loop from gated training to inference validation: execute tinyworlds-training-causal-diagnostics, then tinyworlds-standard-inference-test, inspect inference_results and MSE, and if inference fails automatically retune/retrain dynamics then re-test until pass or retry budget is exhausted. Use when users ask for train-to-inference automation, checkpoint readiness validation, or fixing train-pass but inference-fail issues."
---

# TinyWorlds Train-To-Inference

## Goal
Use one deterministic loop:
1. Run full gated training in order (`video_tokenizer -> latent_actions -> dynamics`).
2. Run standard inference validation (`tinyworlds-standard-inference-test`).
3. Check `inference_results` + metric gate.
4. If fail, retune/retrain `dynamics` and re-run inference.
5. If pass, write a retrospective report under `docs/action/`.

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
  --video-checkpoint results/<run>/video_tokenizer/checkpoints \
  --latent-checkpoint results/<run>/latent_actions/checkpoints \
  --dynamics-checkpoint results/<run>/dynamics/checkpoints \
  --video-chunk 200 \
  --latent-chunk 200 \
  --dynamics-chunk 200 \
  --target-updates 200 \
  --init-log-interval 100 \
  --max-mse 0.03 \
  --max-inference-retries 2
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
- Retrospective report (required):
  - `docs/action/train-to-inference-retrospective-<timestamp>.md`
- Training reports from controller:
  - `docs/action/auto-training-loop-report-<timestamp>.md`
- Inference outputs:
  - `inference_results/*.png`

## Resources
- Gated training controller:
  - `docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py`
- Standard inference entrypoint:
  - `scripts/run_inference.py`
- This skill automation wrapper:
  - `docs/skills/tinyworlds-train-to-inference/scripts/train_to_inference_loop.py`
