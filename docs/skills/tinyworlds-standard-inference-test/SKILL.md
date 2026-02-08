---
name: tinyworlds-standard-inference-test
description: Run a standard inference smoke/regression test with scripts/run_inference.py for a specific TinyWorlds run root, resolve latest checkpoints for video_tokenizer/latent_actions/dynamics, execute consistent overrides, and verify outputs and quality metrics. Use when users ask to validate inference readiness for a run like results/... or debug run_inference checkpoint/path issues.
---

# Tinyworlds Standard Inference Test

## Goal

Use `scripts/run_inference.py` as the standard test entrypoint.

Produce a repeatable test that answers:
- can this run load all 3 models correctly
- can it generate frames end to end
- are core outputs (`png`) and key metrics printed

## Step 1: Prepare Environment

Run from repo root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="$pwd;$env:PYTHONPATH"
```

## Step 2: Resolve Latest Checkpoints Inside One Run Root

Use `utils.find_latest_checkpoint(..., run_root_dir=...)` so the test does not accidentally pick other runs:

```powershell
@'
import os
from utils.utils import find_latest_checkpoint
base = os.getcwd()
run_root = os.path.abspath(r"results/zzzz_zelda_manual_20260208_001")
for model in ["video_tokenizer", "latent_actions", "dynamics"]:
    print(model, find_latest_checkpoint(base, model, run_root_dir=run_root))
'@ | .\.venv\Scripts\python.exe -
```

## Step 3: Run Standard Smoke Test

Use this profile for fast and repeatable validation:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/inference.yaml -- `
  use_latest_checkpoints=false `
  dataset=ZELDA `
  device=cuda `
  use_actions=true `
  use_gt_actions=false `
  use_interactive_mode=false `
  teacher_forced=false `
  context_window=2 `
  prediction_horizon=1 `
  generation_steps=12 `
  preload_ratio=0.02 `
  video_tokenizer_path="<video_tokenizer_ckpt_path>" `
  latent_actions_path="<latent_actions_ckpt_path>" `
  dynamics_path="<dynamics_ckpt_path>"
```

For longer regression runs:
- increase `generation_steps`
- increase `preload_ratio` (for broader sample coverage)

## Step 4: Pass/Fail Criteria

Treat as pass when:
- log prints all 3 selected checkpoints
- loop prints `Inferring frame ...`
- terminal prints `Inference stats`
- `inference_results/inference_results_gt_vs_pred_*.png` exists

Treat as warning when:
- MP4 write fails due `openh264` library
- PNG and inference stats are still produced

Treat as fail when:
- any checkpoint path cannot be loaded
- crash before first inference step

## Step 5: Tested Example (This Repo)

Run root:
- `results/zzzz_zelda_manual_20260208_001`

Resolved checkpoints:
- `video_tokenizer_step_37500`
- `latent_actions_step_9500`
- `dynamics_step_92000`

Observed result:
- inference completed with `Total frames generated: 14`
- example metric: `Mean Squared Error (GT vs Pred): 0.022317`
- PNG visualization generated under `inference_results/`
