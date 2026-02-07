# TinyWorlds Training Diagnostic Checklist

## Scope
Use this checklist for Zelda/TinyWorlds 3-stage training diagnostics:
- Video Tokenizer
- Latent Actions
- Dynamics

## Stage Expectations

### Video Tokenizer
Healthy signs:
- Loss declines strongly from early training.
- Final loss remains low (typically far below 0.01 for this config).
- Reconstructions preserve scene geometry and major object placement.

Warning signs:
- Late-stage reconstructions become noisy or collapse into repeated patterns.
- Codebook usage remains extremely low for most of training.

### Latent Actions
Healthy signs:
- Loss drops substantially from step 0.
- `unique_actions` > 1 and non-zero `action_entropy` appear through training.
- Reconstructions keep coarse scene structure.

Warning signs:
- `unique_actions` locks at 1 for long intervals.
- `action_entropy` is consistently near 0.

### Dynamics
Healthy signs:
- Loss stabilizes or improves in later windows.
- Visual predictions retain scene structure and temporal plausibility.

Critical signs:
- Late loss windows trend upward and stay high.
- Visual outputs collapse into flat color blocks, stripes, or near-constant textures.

## Checkpoint Integrity
For each stage checkpoint directory:
1. Confirm `model_state_dict.pt`, `optim_state_dict.pt`, and `state.pt` exist.
2. Confirm model/optimizer files are not tiny placeholders.
3. Confirm `torch.load()` returns non-empty dicts where expected.

If latest checkpoint is invalid, choose newest valid checkpoint as fallback and report it explicitly.

## Verdict Rules
- `normal`: Numeric trends and visuals match expected stage behavior; latest checkpoint is usable.
- `watch`: Training still runs, but trends/visuals indicate instability or degradation.
- `critical`: Strong collapse indicators or unusable latest checkpoints.

## Reporting Template
Include these sections:
1. Current run status (process alive, stage, latest step/time)
2. Numeric summary by stage
3. Visualization findings by stage
4. Checkpoint integrity summary
5. Final verdict and prioritized next actions
