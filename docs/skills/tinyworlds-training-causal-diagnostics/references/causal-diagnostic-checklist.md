# TinyWorlds Causal Diagnostic Checklist

## Goal
Disentangle whether dynamics degradation is mainly:
- Upstream latent-actions quality bottleneck.
- Dynamics optimization/configuration instability.
- Mixed.

## Latent-Actions Causal Signals
Treat latent as **primary** only if most conditions hold:
1. `unique_actions` spends long periods at `1`.
2. `action_entropy` remains near `0`.
3. `latent_actions/codebook_usage` is persistently very low and shrinking.
4. Visuals lose scene structure in late latent checkpoints.

Treat latent as **secondary** when:
1. Loss converges.
2. Multiple actions remain active (`unique_actions > 1`).
3. Entropy is non-zero but trending down.

## Dynamics-Self Causal Signals
Treat dynamics as **primary** when most conditions hold:
1. Early/mid/late loss windows trend upward.
2. First-10% to last-10% loss delta is strongly positive.
3. `loss` has weak correlation with action diversity metrics.
4. `loss` has moderate/strong correlation with LR schedule or optimizer regime.
5. Visual outputs drift toward flat blocks/striping/low-detail texture.

## Checkpoint Integrity (DDP/FSDP Runs)
Always verify:
1. Files exist: `model_state_dict.pt`, `optim_state_dict.pt`, `state.pt`.
2. Model/optimizer files are not tiny placeholders.
3. `torch.load()` returns non-empty dicts.
4. Newest resumable checkpoint is explicitly identified.

If newest checkpoint is invalid:
- Mark run status at least `watch`.
- If invalid ratio is high or latest unusable, mark `critical`.

## Dual-GPU Configuration Recommendations
When keeping DDP (`use_ddp=true`, `nproc_per_node=2`):
1. Lower dynamics LR before increasing model size.
2. Reduce gradient accumulation if optimizer feedback is too delayed.
3. Keep batch-size increases conditional on memory headroom.
4. Use shorter control horizons (`n_updates`) with periodic re-evaluation.
5. Enable compile only after verifying numerical stability and checkpoint correctness.

## Reporting Minimum
Include:
1. Exact evidence paths.
2. Numeric values for each claim.
3. Causal verdict: latent primary/secondary + dynamics primary/secondary.
4. Immediate fallback checkpoint.
5. Config changes split into:
   - `must change now`
   - `next cycle`
