## Plan: Tokenizer Recon A/B Test

Add an A/B tokenizer reconstruction check right after `context_frames` is created in [scripts/run_inference.py](scripts/run_inference.py), using the existing `_to_vis(...)` and `_save_tokenizer_recon_visualization(...)` patterns. Run two paths on the same `context_frames`: A = original, B = mapped to $[0,1]$ when needed, then `tokenize → get_latents_from_indices → detokenize` for each. For each path, print min/max/mean and Recon MSE, and save a 2xT GT vs Recon PNG with distinct filenames (A vs B) in `inference_results/`. Wrap the whole A/B block in `torch.inference_mode()` and `try/except` to avoid interrupting inference. This keeps the main inference flow unchanged and localizes the test to the existing sanity-check insertion point.

**Steps**
1. Extend helper utilities in [scripts/run_inference.py](scripts/run_inference.py) to support A/B reporting: add a small stats printer and allow `_save_tokenizer_recon_visualization(...)` to accept a label to include in the filename and title.
2. Insert the A/B test block immediately after `context_frames` / `generated_frames` assignment in [scripts/run_inference.py](scripts/run_inference.py#L183-L215), reusing `video_tokenizer.tokenize(...)`, `quantizer.get_latents_from_indices(...)`, and `video_tokenizer.detokenize(...)`.
3. Implement A/B inputs: A uses `context_frames` as-is; B uses a mapped-to-$[0,1]$ tensor (only when values indicate $[-1,1]$), then run recon for both, compute `min/max/mean` on GT and recon, MSE, and save PNGs with suffixes like `_A` and `_B`.
4. Ensure `video_tokenizer.eval()` and device alignment are respected, and keep all outputs non-blocking (guarded by `try/except` with clear error message).

**Verification**
- Run `python scripts/run_inference.py --config configs/inference.yaml`.
- Confirm console logs include stats and MSE for both A and B.
- Confirm two PNGs saved in `inference_results/` with A/B labels.
- Confirm downstream inference outputs still generate (GT vs Pred PNG and MP4).

**Decisions**
- Keep A/B test localized in `run_inference.py` (no refactor into `utils/`) to satisfy scope limits.
- Reuse existing adaptive visualization normalization rather than enforcing a new global assumption.
