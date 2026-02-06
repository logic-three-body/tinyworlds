# WSL Training Errors (2026-02-05)

This note summarizes the issues encountered while restarting training under WSL and how each was resolved.

## 1) torchrun not found
- Symptom: `torchrun: command not found` when invoking training.
- Cause: `torchrun` existed only inside the venv and the PATH did not include it.
- Fix: Run with the venv path or use the venv python entrypoint:
  - `PATH=/root/tinyworlds/.venv/bin:$PATH` and `PYTHONPATH=/root/tinyworlds`
  - Or call `./.venv/bin/torchrun` directly.

## 2) TabError in config file
- Symptom: `TabError: inconsistent use of tabs and spaces in indentation` in `utils/config.py`.
- Cause: Mixed tabs and spaces.
- Fix: Normalize indentation to spaces.

## 3) Python 3.8 typing incompatibility
- Symptom: `TypeError: unsupported operand type(s) for |: 'type' and 'type'`.
- Cause: `X | Y` union syntax requires Python 3.10+.
- Fix: Replace with `typing.Union` and `Optional`.

## 4) OmegaConf ValidationError (NoneType)
- Symptom: `Unexpected type annotation: NoneType` for `offload_policy`.
- Cause: `CPUOffloadPolicy` missing in this build, leaving a NoneType annotation.
- Fix: Use a fallback type when FSDP is unavailable.

## 5) FSDP import errors
- Symptom: `cannot import name 'fully_shard'` or `FSDPModule` from `torch.distributed.fsdp`.
- Cause: FSDP not available in this PyTorch build.
- Fix: Guard FSDP imports and only use FSDP if available.

## 6) SIGKILL from OOM killer
- Symptom: `exitcode -9 (SIGKILL)` during/after data load.
- Cause: System OOM killer (host RAM), confirmed by `dmesg` OOM logs.
- Fixes applied:
  - Reduced `preload_ratio` from 1.0 to 0.5 in `configs/training.yaml`.
  - Reduced `batch_size_per_gpu` in `configs/latent_actions.yaml`.

## 7) torch.compile build failure
- Symptom: `fatal error: Python.h: No such file or directory` from torch inductor.
- Cause: Missing Python development headers.
- Fix: Disable `compile` in `configs/training.yaml` (set `compile: false`).

## 8) FSDPModule isinstance TypeError
- Symptom: `TypeError: isinstance() arg 2 must be a type or tuple of types` in `save_training_state`.
- Cause: `FSDPModule` unavailable, used directly in an `isinstance` tuple.
- Fix: Use a safe tuple that is empty when FSDP is unavailable.

## Current status
- Training starts successfully in the Latent Actions stage after the above fixes.
