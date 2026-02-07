# TinyWorlds Auto-Training Gates

## Goal
Define a deterministic loop for:
1. Train to gate.
2. Diagnose.
3. Continue or retune.
4. Repeat until pass.

## Default Gate Schedule

## Video Tokenizer
- Chunk size: `5000` updates
- Target: `40000`
- Pass gate when:
  - loss trend is downward over last 2 gates
  - latest `train/loss <= 0.01`
  - latest checkpoint is valid

## Latent Actions
- Chunk size: `1000` updates
- Target: `10000` (or user override)
- Pass gate when:
  - latest `train/loss <= 0.06`
  - latest `unique_actions >= 2`
  - latest `action_entropy >= 0.4`
  - latest checkpoint is valid

## Dynamics
- Chunk size: `5000` updates
- Target: user configured horizon (default often long)
- Pass gate when:
  - late-window mean loss is not increasing versus mid-window
  - first10%-last10% delta is non-positive or near flat
  - visuals keep scene structure (no persistent block/stripe collapse)
  - latest checkpoint is valid

## Retry Budget
- Per stage max retries: `5` (suggested default)
- If retries exceeded:
  - stop stage
  - report blocking factors and best known checkpoint

## Retune Ladders
Apply one rung per failed gate, then rerun next chunk.

## Video Tokenizer Ladder
1. `learning_rate *= 0.5`
2. `gradient_accumulation_steps += 1`
3. `log_interval` reduce to observe faster

## Latent Actions Ladder
1. `learning_rate *= 0.8`
2. increase `n_updates` target by one chunk if loss still dropping
3. if entropy collapse: reduce `n_actions` in next full-cycle retrain only

## Dynamics Ladder
1. `learning_rate *= 0.2`
2. `gradient_accumulation_steps` reduce by 1 (min 1)
3. `log_interval` reduce (e.g., 2000 -> 1000)
4. shorten current run horizon and re-evaluate earlier

Keep DDP enabled unless user explicitly allows single-GPU fallback.

## Checkpoint Acceptance
For each candidate checkpoint:
1. files exist (`model_state_dict.pt`, `optim_state_dict.pt`, `state.pt`)
2. model/optimizer file size is not tiny placeholder
3. `torch.load()` gives non-empty dict

Accept newest valid checkpoint as `resume_from`.

## Decision Output Format
For each gate, output:
1. `stage`
2. `gate_step`
3. `decision` (`continue|retune|restart_stage|stop`)
4. `metrics` (key values)
5. `resume_from`
6. `config_patch_applied`
