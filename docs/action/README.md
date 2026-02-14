# Action Docs Policy

This directory contains operation notes, guides, and generated analysis artifacts.

## Output Location Rule

For all new training/inference outputs, follow `docs/data-output-governance.md`:

- Default root: `/mnt/y/WorldModel/Tinyworld_backup/<YYYYMMDD_HHMMSS>/`
- Keep structure under that root:
  - `results/...`
  - `inference_results/...`
  - `wandb/run-*/...`
  - `docs/action/...` (generated reports/plots)

## Temporary Local Logs (Not Versioned)

The following migration logs are temporary local files and must stay out of git history:

- `docs/action/asset-move-report-*.md`
- `docs/action/asset-move-manifest-*.txt`

They are local operation evidence only.
