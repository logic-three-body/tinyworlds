import torch
from datasets.data_utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob
import re
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint, load_dynamics_from_checkpoint, find_latest_checkpoint
from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models, visualize_inference, sample_random_action, get_action_latent
from einops import repeat
from typing import Optional
from datetime import datetime


def _to_vis(frames):
    """
    Convert frames to [0, 1] range with adaptive normalization.
    
    Args:
        frames: tensor of shape [B, T, C, H, W] or similar
    
    Returns:
        float tensor in [0, 1] on CPU
    """
    frames = frames.float()
    
    # Detect value range adaptively
    min_val = frames.min().item()
    max_val = frames.max().item()
    
    # Normalize based on detected range
    if min_val < 0:
        # Likely [-1, 1] range
        frames = (frames + 1.0) / 2.0
    elif max_val > 1:
        # Likely [0, 255] range
        frames = frames / 255.0
    # else: already in [0, 1]
    
    # Clamp to [0, 1]
    frames = torch.clamp(frames, 0, 1)
    
    return frames.detach().cpu()


def _map_to_unit_range(frames):
    frames = frames.float()
    min_val = frames.min().item()
    max_val = frames.max().item()
    if min_val < 0:
        return (frames + 1.0) / 2.0
    if max_val > 1:
        return frames / 255.0
    return frames


def _print_frame_stats(label, frames):
    frames = frames.float()
    min_val = frames.min().item()
    max_val = frames.max().item()
    mean_val = frames.mean().item()
    print(f"{label} min/max/mean: {min_val:.6f} / {max_val:.6f} / {mean_val:.6f}")


def _save_tokenizer_recon_visualization(gt_frames, recon_frames, mse, label):
    """
    Save GT vs Recon comparison visualization.
    
    Args:
        gt_frames: ground truth frames [B, T, C, H, W]
        recon_frames: reconstructed frames [B, T, C, H, W]
        mse: MSE value (float)
    """
    # Normalize both to [0, 1]
    gt_vis = _to_vis(gt_frames)
    recon_vis = _to_vis(recon_frames)
    
    # Extract first batch
    gt_vis = gt_vis[0]  # [T, C, H, W]
    recon_vis = recon_vis[0]  # [T, C, H, W]
    
    T = gt_vis.shape[0]
    
    # Create figure with 2 rows x T columns
    fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
    
    # Handle case where T == 1 (axes is 1D)
    if T == 1:
        axes = axes.reshape(2, 1)
    
    for t in range(T):
        # Convert to numpy and transpose to [H, W, C]
        gt_img = gt_vis[t].permute(1, 2, 0).numpy()
        recon_img = recon_vis[t].permute(1, 2, 0).numpy()
        
        # Handle grayscale case
        if gt_img.shape[2] == 1:
            gt_img = gt_img.squeeze(-1)
            recon_img = recon_img.squeeze(-1)
            cmap = 'gray'
        else:
            # Take first 3 channels if more than 3
            if gt_img.shape[2] > 3:
                gt_img = gt_img[..., :3]
                recon_img = recon_img[..., :3]
            cmap = None
        
        # Plot GT (top row)
        axes[0, t].imshow(gt_img, cmap=cmap)
        axes[0, t].set_title(f"GT t={t}", fontsize=8)
        axes[0, t].axis('off')
        
        # Plot Recon (bottom row)
        axes[1, t].imshow(recon_img, cmap=cmap)
        axes[1, t].set_title(f"Recon t={t}", fontsize=8)
        axes[1, t].axis('off')
    
    # Add overall title with MSE
    fig.suptitle(f"Tokenizer Recon {label}: MSE = {mse:.6f}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs('inference_results', exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'inference_results/tokenizer_recon_gt_vs_recon_{label}_{timestamp}.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Tokenizer recon visualization saved to: {save_path}")


def main():
    # load inference config
    args: InferenceConfig = load_config(InferenceConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml'))

    # enable tf32 if requested
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # whether any setting requires using action tokens
    use_latent_actions = (args.use_actions or args.use_gt_actions or args.use_interactive_mode)

    # check if any path is missing
    def missing(path: Optional[str]) -> bool:
        return (path is None) or (not os.path.isfile(path))

    # resolve latest checkpoints if requested or any path missing
    base_dir = os.getcwd()
    if args.use_latest_checkpoints or missing(args.video_tokenizer_path):
        vt_ckpt = find_latest_checkpoint(base_dir, "video_tokenizer")
        args.video_tokenizer_path = vt_ckpt
    if (args.use_latest_checkpoints or missing(args.latent_actions_path)) and use_latent_actions:
        lam_ckpt = find_latest_checkpoint(base_dir, "latent_actions")
        args.latent_actions_path = lam_ckpt
    if args.use_latest_checkpoints or missing(args.dynamics_path):
        dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")
        args.dynamics_path = dyn_ckpt
    
    # confirm which ckpts are being used
    print(f"Using video_tokenizer checkpoint: {args.video_tokenizer_path}")
    if use_latent_actions:
        print(f"Using latent_actions checkpoint: {args.latent_actions_path}")
    print(f"Using dynamics checkpoint: {args.dynamics_path}")

    # validate required paths
    if missing(args.video_tokenizer_path):
        raise FileNotFoundError("video_tokenizer_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints with available runs.")
    if use_latent_actions and missing(args.latent_actions_path):
        raise FileNotFoundError("latent_actions_path is not set or not a file while actions are requested. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
    if missing(args.dynamics_path):
        raise FileNotFoundError("dynamics_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
 
    # load models, optionally compile
    video_tokenizer, latent_action_model, dynamics_model = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, args.device, use_actions=use_latent_actions)
    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead", fullgraph=False, dynamic=True)
        if use_latent_actions:
            latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Compiled all models for inference.")

    # determine how many ground-truth frames we need in each batch: context + generation steps + prediction horizon
    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon

    # dataloader
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides = {'preload_ratio': args.preload_ratio}
    else:
        data_overrides = {}
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=frames_to_load, **data_overrides)

    # sample random batch
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    og_ground_truth_frames = data_loader.dataset[random_idx][0]  # full sequence
    og_ground_truth_frames = og_ground_truth_frames.unsqueeze(0).to(args.device)  # [1, frames_to_load, C, H, W]

    ground_truth_frames = og_ground_truth_frames[:, :frames_to_load, :, :, :]  # [1, frames_to_load, C, H, W]

    # start with initial context (first context_window GT frames)
    context_frames = ground_truth_frames[:, :args.context_window, :, :, :]
    generated_frames = context_frames.clone()

    # === Tokenizer Recon A/B Test ===
    print("\n=== Tokenizer Recon A/B Test ===")
    try:
        video_tokenizer.eval()
        with torch.inference_mode():
            test_inputs = {
                "A": context_frames,
                "B": torch.clamp(_map_to_unit_range(context_frames), 0.0, 1.0),
            }

            for label, test_frames in test_inputs.items():
                idx = video_tokenizer.tokenize(test_frames)
                lat = video_tokenizer.quantizer.get_latents_from_indices(idx)
                recon = video_tokenizer.detokenize(lat)

                mse = torch.mean((recon.float() - test_frames.float()) ** 2).item()
                print(f"[{label}] Recon MSE: {mse:.6f}")
                _print_frame_stats(f"[{label}] Input", test_frames)
                _print_frame_stats(f"[{label}] Recon", recon)

                _save_tokenizer_recon_visualization(test_frames, recon, mse, label)

        print("=== A/B Test Complete ===\n")
    except Exception as e:
        print(f"[ERROR] Tokenizer recon A/B test failed: {e}")
        print("Continuing with inference despite the error.\n")

    # initialize actions
    n_actions = None
    inferred_actions = []
    if use_latent_actions:
        n_actions = latent_action_model.quantizer.codebook_size        

    # ensure we don’t exceed available GT frames if in teacher-forced mode
    max_possible_steps = ground_truth_frames.shape[1] - args.context_window
    if args.teacher_forced and args.generation_steps > max_possible_steps:
        print(f"[WARN] Requested {args.generation_steps} generation steps but only {max_possible_steps} are possible with teacher-forced context. Clamping.")
    effective_steps = args.generation_steps if not args.teacher_forced else min(args.generation_steps, max_possible_steps)

    for i in range(effective_steps):
        print(f"Inferring frame {i+1}/{effective_steps}")
        # select context depending on teacher-forced flag
        if args.teacher_forced:
            context_start = i  # shift window along ground truth
            context_frames = ground_truth_frames[:, context_start:context_start+args.context_window, :, :, :] # [1, context_window, C, H, W]
        else:
            # autoregressive: last context_window frames from generated sequence
            context_frames = generated_frames[:, -args.context_window:, :, :, :]  # [1, context_window, C, H, W]

        # encode context frames each iteration
        video_indices = video_tokenizer.tokenize(context_frames)
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

        sampled_action_index, action_latent = get_action_latent(args, inferred_actions, n_actions, context_frames, latent_action_model, i)

        # dynamics forward inference needs idx -> latents fun
        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        # autocast for inference if amp enabled (bfloat16 on CUDA by default)
        autocast_dtype = torch.bfloat16 if args.amp else None
        with torch.amp.autocast('cuda', enabled=args.amp, dtype=autocast_dtype):
            next_video_latents = dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=args.prediction_horizon,
                num_steps=10,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=args.temperature,
            )

        # decode next video tokens to frames
        next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]

        generated_frames = torch.cat([generated_frames, next_frames[:, -args.prediction_horizon:, :, :]], dim=1)
        # TODO: if using interactive mode, visualize next_frames[:, -1] (recently inferred frame) every time, probably with matplotlib is easiest
        # point is for user to be able to interact with it in real time

    # visualize inference
    visualize_inference(generated_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=use_latent_actions)


if __name__ == "__main__":
    main()
