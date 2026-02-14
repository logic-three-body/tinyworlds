---
name: tinyworlds-training-diagnostics
description: Diagnose TinyWorlds training health from local artifacts and detect drift/anomalies across video tokenizer, latent actions, and dynamics. Use when users ask to analyze current training status, export latest metrics to Logs, compare runs with results visualizations, validate checkpoints, or determine whether training has deviated from normal behavior.
---

# TinyWorlds Training Diagnostics

## Overview
Use a repeatable workflow to:
1. Export latest metrics from local `.wandb` transaction files into `Logs/`.
2. Evaluate numeric trends for each stage (`video_tokenizer`, `latent_actions`, `dynamics`).
3. Cross-check with governed `results/*/visualizations` images.
4. Verify checkpoint integrity and resumability.
5. Output a concise verdict: `normal`, `watch`, or `critical`.

## Output Governance
Use `docs/data-output-governance.md` as source of truth.

Define:
- `RUN_BUNDLE_ROOT=/mnt/y/WorldModel/Tinyworld_backup/<YYYYMMDD_HHMMSS>`

When collecting evidence, prefer these locations:
- `RUN_BUNDLE_ROOT/results/...`
- `RUN_BUNDLE_ROOT/wandb/run-*/...`
- `RUN_BUNDLE_ROOT/docs/action/...` (generated analysis outputs)

Temporary migration logs are local-only and git-ignored:
- `docs/action/asset-move-report-*.md`
- `docs/action/asset-move-manifest-*.txt`

## Workflow

### Step 1: Collect Run Context
Read only what is needed:
- `README.md` for expected pipeline behavior (tokenizer -> action tokenizer -> dynamics).
- Target notes under `docs/action/` (user-provided context files).
- `RUN_BUNDLE_ROOT/results/` directory tree and latest timestamps.
- `RUN_BUNDLE_ROOT/wandb/run-*/` local run folders.

### Step 2: Export Latest Metrics to Logs
Preferred:
- If W&B API is reachable, export run history via API.

Fallback (common on restricted servers):
- Decode local `.wandb` files with `scripts/export_local_wandb_history.py`.
- Output files to `Logs/` using suffix `latest`.

Example:
```bash
python docs/skills/tinyworlds-training-diagnostics/scripts/export_local_wandb_history.py \
  --run "video_tokenizer=wandb/run-20260204_195233-okmjny24/run-okmjny24.wandb" \
  --run "latent_actions=wandb/run-20260205_210649-d376skck/run-d376skck.wandb" \
  --run "dynamics=wandb/run-20260206_004434-2pghxx71/run-2pghxx71.wandb" \
  --out-dir Logs \
  --suffix latest
```

### Step 3: Run Numeric Diagnostics
Compute at least:
- Stage progress (`step_max`, row count, final loss, min/median loss).
- Trend windows (early/mid/late means) for `dynamics` loss.
- Action stats (`unique_actions`, `action_entropy`) for action-conditioned stages.
- Throughput from `_runtime` and/or checkpoint timestamps.

See thresholds and interpretation rules in `references/diagnostic-checklist.md`.

### Step 4: Visual Diagnostics
Inspect representative images from each stage:
- `RUN_BUNDLE_ROOT/results/<run>/video_tokenizer/visualizations/*`
- `RUN_BUNDLE_ROOT/results/<run>/latent_actions/visualizations/*`
- `RUN_BUNDLE_ROOT/results/<run>/dynamics/visualizations/*`

Compare early vs late outputs:
- Improvement toward structure-preserving reconstructions is expected.
- Flat color blocks, repetitive striping, or near-constant output in late `dynamics` is a strong anomaly signal.

### Step 5: Checkpoint Integrity
For each checkpoint folder, verify:
- `model_state_dict.pt` and `optim_state_dict.pt` are not tiny placeholders.
- `torch.load()` returns non-empty dicts where expected.
- Latest checkpoint is actually resumable.

If many checkpoints are empty while training continues, report as `critical` even if process is alive.

### Step 6: Produce Final Verdict
Use three-level status:
- `normal`: metrics and visuals align with expected training behavior.
- `watch`: training runs but shows degradation or instability that may still recover.
- `critical`: clear collapse or unusable checkpoints / severe divergence.

When writing conclusions, always include:
- Exact file evidence paths.
- Concrete metric values (not only qualitative words).
- Immediate safe fallback checkpoint if latest is unusable.
- Next actions in priority order.

## Resources
- Export helper: `scripts/export_local_wandb_history.py`
- Checklist and thresholds: `references/diagnostic-checklist.md`
