# TinyWorlds Data Output Governance

## Purpose

This document is the single source of truth for TinyWorlds data output locations.
Starting from now, new application/training/inference artifact outputs default to the Y drive backup root used in the latest migration.

## Default Output Root (Mandatory)

- Windows path: `Y:\WorldModel\Tinyworld_backup`
- WSL path: `/mnt/y/WorldModel/Tinyworld_backup`

All new output data should be written under this root by default.

## Output Layout Standard

Use timestamp subdirectories to avoid overwrite:

- Run root pattern: `<OUTPUT_ROOT>/<YYYYMMDD_HHMMSS>/`
- Keep relative structure under run root:
  - `results/...`
  - `inference_results/...`
  - `wandb/run-.../...`
  - `docs/action/...` (generated analysis artifacts only)

Example:

- `/mnt/y/WorldModel/Tinyworld_backup/20260213_021236/results/...`
- `/mnt/y/WorldModel/Tinyworld_backup/20260213_021236/inference_results/...`

## Scope Rules

Move/store as data artifacts:

- `results/**`
- `inference_results/**`
- `wandb/run-*/**`
- generated analysis outputs in `docs/action/**` (reports/csv/json/plots)

Do not treat as output artifacts:

- `.venv/**`, `.vscode/**`, `**/__pycache__/**`
- `data/**` (unless explicitly requested)
- config/script files (for example `*.yaml`, `*.ps1`, `PROXY_GUIDE.md`)
- logs that are environment/runtime diagnostics (for example `*.log`, `Logs/**`, `wandb/debug*.log`)

## Existing Migration Reference

Latest completed migration evidence:

- Report: `docs/action/asset-move-report-20260213_021236.md`
- Manifest: `docs/action/asset-move-manifest-20260213_021236.txt`
- Target root used: `/mnt/y/WorldModel/Tinyworld_backup/20260213_021236`

## Operational Requirements

- When running from WSL, ensure Y drive is mounted:
  - `mount -t drvfs Y: /mnt/y`
- If a script supports output path override, set it to `/mnt/y/WorldModel/Tinyworld_backup/<timestamp>/...`.
- If a script does not support override, generate locally then migrate using the asset move workflow and record a report in `docs/action/`.

## Change Control

- Any change to default output root must update this file first.
- Any large output migration must append a new `docs/action/asset-move-report-<timestamp>.md` and matching manifest.
