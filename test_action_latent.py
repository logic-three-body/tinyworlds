#!/usr/bin/env python3
"""Quick test of action latent creation."""

import sys
import os
import torch
from einops import repeat

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint

# Load config
args = load_config(InferenceConfig, default_config_path='configs/inference.yaml')
args.dataset = "POLE_POSITION"
args.use_actions = True
args.context_window = 2
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find latest pole_position run
results_dir = 'results'
pole_runs = sorted([d for d in os.listdir(results_dir) if 'pole_position' in d.lower()], reverse=True)
latest_pole_run = os.path.join(results_dir, pole_runs[0])

args.video_tokenizer_path = find_latest_checkpoint('.', "video_tokenizer", run_root_dir=latest_pole_run)
args.latent_actions_path = find_latest_checkpoint('.', "latent_actions", run_root_dir=latest_pole_run)
args.dynamics_path = find_latest_checkpoint('.', "dynamics", run_root_dir=latest_pole_run)

# Load latent action model
_, latent_action_model, _ = load_models(
    args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path,
    args.device, use_actions=True
)
latent_action_model.eval()

print("Testing action latent creation:")
print(f"Codebook size: {latent_action_model.quantizer.codebook_size}")

# Test creating action latent like in the script
action_history = [0]  # Initial action
action_id = 5

# Simulate the _get_action_latent method
recent_actions = action_history[-args.context_window:]
if len(recent_actions) < args.context_window:
    pad_size = args.context_window - len(recent_actions)
    pad_actions = [0] * pad_size
    all_actions = pad_actions + recent_actions
else:
    all_actions = recent_actions

print(f"All actions: {all_actions}")

actions_tensor = torch.tensor(all_actions, device=args.device)
print(f"Actions tensor shape: {actions_tensor.shape}")

try:
    repeated = repeat(actions_tensor, 'i -> 1 i')
    print(f"Repeated tensor shape: {repeated.shape}")
    
    action_latent = latent_action_model.quantizer.get_latents_from_indices(repeated)
    print(f"✓ Action latent created successfully")
    print(f"  Shape: {action_latent.shape}")
    print(f"  Range: [{action_latent.min():.3f}, {action_latent.max():.3f}]")
except Exception as e:
    import traceback
    print(f"✗ Failed to create action latent: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Action latent creation works!")
