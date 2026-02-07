---
name: tinyworlds-train-to-inference
description: Summarize and execute the TinyWorlds end to end workflow from dataset download and prep, through three stage training (video_tokenizer, latent_actions, dynamics), checkpoint validation, offline inference, and interactive play scripts (play_zelda.py, play_sonic.py, play_pole.py). Use when users ask to run from training data to inference, make a newly trained run playable, or debug why trained checkpoints cannot be used for inference.
---

# Tinyworlds Train To Inference

## Overview

Standardize one workflow:
environment setup -> data prep -> training -> checkpoint validation -> inference -> interactive play.

## Step 1: Prepare Environment

Run from repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
```

For CUDA runs, verify:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Step 2: Prepare Dataset

Download dataset files into `data/`:

```powershell
.\.venv\Scripts\python.exe scripts/download_assets.py datasets --pattern "zelda_frames.h5"
```

Common dataset file names:
- `zelda_frames.h5`
- `sonic_frames.h5`
- `pole_position_frames.h5`
- `picodoom_frames.h5`
- `pong_frames.h5`

Verify file exists:

```powershell
Test-Path data\zelda_frames.h5
```

## Step 3: Train Models

Prefer full pipeline:

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
.\.venv\Scripts\python.exe scripts/full_train.py --config configs/training.yaml -- --dataset=ZELDA
```

Run by stage when needed:

```powershell
.\.venv\Scripts\python.exe scripts/train_video_tokenizer.py --config configs/video_tokenizer.yaml --training_config configs/training.yaml
.\.venv\Scripts\python.exe scripts/train_latent_actions.py --config configs/latent_actions.yaml --training_config configs/training.yaml
.\.venv\Scripts\python.exe scripts/train_dynamics.py --config configs/dynamics.yaml --training_config configs/training.yaml video_tokenizer_path=<video_ckpt_dir> latent_actions_path=<latent_ckpt_dir>
```

Notes:
- `full_train.py` chains all stages and writes one run root in `results/<timestamp>/`.
- `train_dynamics.py` requires valid tokenizer and latent action checkpoints.

## Step 4: Validate Checkpoints

Each checkpoint directory should contain:
- `model_state_dict.pt`
- `optim_state_dict.pt`
- `state.pt`

Detect suspicious tiny model weights:

```powershell
Get-ChildItem results -Recurse -Filter model_state_dict.pt | Where-Object { $_.Length -le 1024 } | Select-Object FullName, Length
```

If the latest step is invalid, fall back to the previous valid step.

## Step 5: Run Offline Inference

Run with latest available checkpoints:

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/inference.yaml -- use_latest_checkpoints=true dataset=ZELDA device=cuda use_actions=true
```

Success signals:
- Terminal prints selected checkpoint paths.
- `inference_results/*.png` is generated.

## Step 6: Run Interactive Play

Zelda:

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
.\.venv\Scripts\python.exe scripts/play_zelda.py
```

Sonic:

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
.\.venv\Scripts\python.exe scripts/play_sonic.py
```

Pole Position:

```powershell
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
.\.venv\Scripts\python.exe scripts/play_pole.py
```

## Step 7: Fix "Trained But Not Playable"

Apply this checklist:

1. Ensure all three checkpoints are valid and loadable.
2. If checkpoints come from different runs, create one merged run folder.
3. Match run name suffix to play script selector:
- include `zelda` for `play_zelda.py`
- include `sonic` for `play_sonic.py`
- include `pole` or `pole_position` for `play_pole.py`
4. Prefer latest valid step, not just latest step number.

Use this merged layout:

```text
results/<timestamp>_zelda/
  video_tokenizer/checkpoints/<video_tokenizer_step_xxx>/
  latent_actions/checkpoints/<latent_actions_step_xxx>/
  dynamics/checkpoints/<dynamics_step_xxx>/
```

## Quick Template: Data To Playable Zelda

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
python scripts/download_assets.py datasets --pattern "zelda_frames.h5"
python scripts/full_train.py --config configs/training.yaml -- --dataset=ZELDA
python scripts/run_inference.py --config configs/inference.yaml -- use_latest_checkpoints=true dataset=ZELDA device=cuda use_actions=true
python scripts/play_zelda.py
```
