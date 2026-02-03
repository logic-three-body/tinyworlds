#!/usr/bin/env python3
"""
Quick test to verify play_pole.py can initialize without errors.
"""

import sys
import os

# Setup path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint
from datasets.data_utils import load_data_and_data_loaders
import torch

print("Testing Pole Position game initialization...")

# Load config
args = load_config(
    InferenceConfig,
    default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml')
)

args.dataset = "POLE_POSITION"
args.use_actions = True
args.prediction_horizon = 1
args.context_window = 2

# Find latest pole_position run
base_dir = os.getcwd()
results_dir = os.path.join(base_dir, 'results')

pole_runs = sorted(
    [d for d in os.listdir(results_dir) if 'pole_position' in d.lower()],
    reverse=True
)

if pole_runs:
    latest_pole_run = os.path.join(results_dir, pole_runs[0])
    print(f"✓ Found Pole Position run: {pole_runs[0]}")
    
    args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer", run_root_dir=latest_pole_run)
    args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions", run_root_dir=latest_pole_run)
    args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics", run_root_dir=latest_pole_run)
else:
    print("✗ No Pole Position checkpoints found")
    sys.exit(1)

print(f"✓ Video tokenizer: {os.path.basename(args.video_tokenizer_path)}")
print(f"✓ Latent actions: {os.path.basename(args.latent_actions_path)}")
print(f"✓ Dynamics: {os.path.basename(args.dynamics_path)}")

# Load models
print("\nLoading models...")
try:
    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path,
        args.latent_actions_path,
        args.dynamics_path,
        args.device,
        use_actions=True
    )
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load models: {e}")
    sys.exit(1)

# Load dataset
print("\nLoading dataset...")
try:
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=args.context_window + 1)
    print(f"✓ Dataset loaded: {len(data_loader.dataset)} samples")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

# Test basic inference
print("\nTesting basic inference step...")
try:
    video_tokenizer.eval()
    latent_action_model.eval()
    dynamics_model.eval()
    
    # Get a sample
    gt_frames = data_loader.dataset[0][0]
    gt_frames = gt_frames.unsqueeze(0).to(args.device)  # [1, T, C, H, W]
    
    context_frames = gt_frames[:, :args.context_window, :, :, :]
    
    with torch.inference_mode():
        # Tokenize
        video_indices = video_tokenizer.tokenize(context_frames)
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)
        print(f"  • Context latents shape: {video_latents.shape}")
        
        # Create action latent
        action_tensor = torch.tensor([[0, 0]], device=args.device)  # [1, context_window]
        action_latent = latent_action_model.quantizer.get_latents_from_indices(action_tensor)
        print(f"  • Action latent shape: {action_latent.shape}")
        
        # Forward inference
        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)
        
        next_video_latents = dynamics_model.forward_inference(
            context_latents=video_latents,
            prediction_horizon=args.prediction_horizon,
            num_steps=4,
            index_to_latents_fn=idx_to_latents,
            conditioning=action_latent,
            temperature=0.0,
        )
        print(f"  • Next latents shape: {next_video_latents.shape}")
        
        # Detokenize
        next_frames = video_tokenizer.detokenize(next_video_latents)
        next_frames = next_frames.clamp(-1.0, 1.0)
        print(f"  • Next frames shape: {next_frames.shape}, range: [{next_frames.min():.3f}, {next_frames.max():.3f}]")
        
    print("✓ Inference step successful!")
    
except Exception as e:
    import traceback
    print(f"✗ Inference failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! play_pole.py should work correctly.")
