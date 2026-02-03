# Pole Position Dataset Extension Plan

## Overview
Extend the SONIC inference and interactive game implementation to the Pole Position dataset. This plan follows the same workflow pattern established with SONIC (A/B testing, tokenizer validation, clamping, interactive game).

## Prerequisites
- Verify existing SONIC workflow is stable and documented
- Confirm model download infrastructure (huggingface_hub or scripts/download_assets.py)
- Ensure hardware can handle additional dataset (RTX 3060 Ti, 12GB VRAM)

## Step 1: Download Pole Position Data
**Objective:** Acquire `pole_position_frames.h5` dataset file

**Action:**
```bash
# Option A: Using download_assets.py
python scripts/download_assets.py data --suite-name pole_position

# Option B: Manual huggingface_hub download
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='AlmondGod/tinyworlds', filename='pole_position_frames.h5', repo_type='dataset', local_dir='data/')"
```

**Verification:**
- File exists at: `data/pole_position_frames.h5`
- File size: ~249 MB (expected)
- Test load with: `h5py.File('data/pole_position_frames.h5', 'r').keys()`

---

## Step 2: Download Pole Position Pretrained Models
**Objective:** Acquire 3 pretrained models (video_tokenizer, latent_actions, dynamics)

**Models to Download:**
1. `pole_position_video_tokenizer_*.pth`
2. `pole_position_latent_actions_*.pth`
3. `pole_position_dynamics_*.pth`

**Target Location:**
```
results/<timestamp>_pole_position/
├── video/checkpoints/pole_position/
├── latent/checkpoints/pole_position/
└── dynamics/checkpoints/pole_position/
```

**Action:**
```bash
# Option A: Using download_assets.py
python scripts/download_assets.py models --suite-name pole_position

# Option B: Manual download with huggingface_hub
# Download each model individually and organize into results/ structure
```

**Verification:**
- All `.pth` files present in correct directories
- Model files load without corruption: `torch.load('path/to/model.pth')`

---

## Step 3: Smoke Test - Offline Inference
**Objective:** Validate data loading, model initialization, and basic inference pipeline

**Command:**
```bash
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"; python scripts/run_inference.py --config configs/inference.yaml -- use_latest_checkpoints=true dataset=POLE_POSITION device=cuda use_actions=true
```

**Expected Outcomes:**
- ✅ Dataset loads successfully (~X frames from pole_position_frames.h5)
- ✅ All 3 models load and initialize in eval mode
- ✅ 10-frame generation completes without errors
- ✅ Output files created:
  - `inference_results/*/recon_visualization.png`
  - `inference_results/*/generated_frames.mp4`
- ✅ MSE output shows reasonable values (< 0.3 expected)

**Troubleshooting:**
- If dataset not found: Verify `configs/inference.yaml` has `dataset: POLE_POSITION` or adjust as needed
- If model not found: Check `results/_pole_position/` directory structure matches expected layout
- If CUDA OOM: Reduce batch_size or context_window in config

---

## Step 4: Create Interactive Game Script
**Objective:** Implement `scripts/play_pole.py` with real-time inference and keyboard control

**Script Architecture:**
- `PolePositionGameState` class (mirrors `SonicGameState`)
- State management: model, dataset, action history, frame generation
- `step(action_id)`: Single-frame inference loop
- `render()`: OpenCV display with HUD (FPS, action, frame count)
- `_get_action_latent()`: Action conditioning with zero-padding
- Game loop: WASD + 0-9 action input, ESC to exit, ~2-5 FPS target

**Key Features:**
- AMP (Automatic Mixed Precision) enabled for speed
- INTER_NEAREST interpolation for pixel art asset preservation
- Real-time frame rendering at 512×512
- Action ID visualization on HUD
- Frame counter and FPS meter

**Configuration:**
- Dataset: "POLE_POSITION"
- Context window: 2 frames
- n_actions: 9 (pole position action space)
- Device: cuda
- Precision: float16 with AMP

---

## Step 5: Execute Interactive Game
**Objective:** Launch real-time interactive game and validate keyboard responsiveness

**Command:**
```bash
.\.venv\Scripts\Activate.ps1; $env:PYTHONPATH = "$pwd;$env:PYTHONPATH"; python scripts/play_pole.py
```

**Expected Behavior:**
- OpenCV window opens (512×512 display)
- Initial frame displays (random clip from dataset)
- Keyboard input detected (WASD/0-9)
- Frame updates on each keystroke
- FPS counter shows 2-5 FPS (pole position may be slower than sonic)
- ESC key cleanly exits

**Performance Metrics to Log:**
- Initialization time
- Average FPS
- Frame generation time per step
- Memory usage

---

## Step 6: Verification & Cross-Dataset Comparison
**Objective:** Validate Pole Position results against SONIC baseline and document differences

**Comparison Metrics:**

| Metric | SONIC | Pole Position | Notes |
|--------|-------|---------------|-------|
| A/B Test MSE ([-1,1] vs [0,1]) | A: 0.089, B: 0.185 | TBD | Expect similar ratio |
| Detokenize Clamp Required | Yes (max: 1.18) | TBD | Check output bounds |
| Interactive FPS | 2-5 FPS | TBD | May vary with dataset |
| MSE (Inference Step 0) | ~0.074 | TBD | Target < 0.3 |

**Documentation:**
- Update `docs/action/` with Pole Position results
- Note any dataset-specific adjustments (e.g., action space, frame resolution, preprocessing)
- Document inference speed differences
- Add Pole Position quick-start commands

**Expected Findings:**
- Input domain preference should match SONIC ([-1,1])
- Clamping may be required depending on model training
- Pixel art nature may require different interpolation (INTER_NEAREST working hypothesis)
- Frame rate may differ based on dataset complexity

---

## Success Criteria
- ✅ All 3 models downloaded and loadable
- ✅ Smoke test completes without errors
- ✅ `play_pole.py` runs with real-time inference
- ✅ Keyboard input produces visible frame changes
- ✅ No CUDA OOM or crashes during extended play
- ✅ Documentation updated with Pole Position results
- ✅ Cross-dataset comparison table populated

---

## Timeline Estimate
- Step 1-2 (Downloads): 5-10 minutes
- Step 3 (Smoke Test): 2-3 minutes
- Step 4 (Script Creation): 10-15 minutes
- Step 5 (Execution): 5 minutes
- Step 6 (Documentation): 10 minutes

**Total: ~45 minutes for complete workflow**

---

## Notes & Considerations
- Pole Position has 9 actions vs SONIC's 16
- Pixel art rendering may require INTER_NEAREST (preserve pixel grid)
- Expected frame count: ~50K+ frames in H5 (vs SONIC's 41K)
- AMP recommended for speed (lower precision acceptable for faster preview)
- If models not available on HuggingFace yet, manual download may be required
