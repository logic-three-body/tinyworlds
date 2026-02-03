"""
Real-time interactive Pole Position game with keyboard control.
Uses tinyworlds inference pipeline for single-frame prediction.

Controls:
  WASD: mapped actions
  0-8: direct action index (Pole Position has 9 actions)
  ESC: quit
"""

import torch
import cv2
import numpy as np
import os
import random
from typing import Optional

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint
from einops import repeat


class PolePositionGameState:
    """Manages game state and inference for Pole Position."""
    
    def __init__(self, args: InferenceConfig):
        self.args = args
        self.device = args.device
        
        # Load models
        print("Loading models...")
        self.video_tokenizer, self.latent_action_model, self.dynamics_model = load_models(
            args.video_tokenizer_path,
            args.latent_actions_path,
            args.dynamics_path,
            args.device,
            use_actions=True
        )
        
        # Set to eval mode
        self.video_tokenizer.eval()
        self.latent_action_model.eval()
        self.dynamics_model.eval()
        
        # Load dataset and sample initial context
        print("Loading dataset...")
        frames_to_load = args.context_window + 1
        _, _, data_loader, _, _ = load_data_and_data_loaders(
            dataset=args.dataset, batch_size=1, num_frames=frames_to_load)
        
        random_idx = random.randint(0, len(data_loader.dataset) - 1)
        gt_frames = data_loader.dataset[random_idx][0]
        gt_frames = gt_frames.unsqueeze(0).to(args.device)  # [1, T, C, H, W]
        
        # Initialize game state
        self.generated_frames = gt_frames[:, :args.context_window, :, :, :]  # [1, context_window, C, H, W]
        self.action_history = []  # Track last actions for padding
        self.n_actions = self.latent_action_model.quantizer.codebook_size
        self.frame_count = 0
        self.fps_clock = cv2.getTickCount()
        self.current_action_id = 0
        
        print(f"Initialized with context_window={args.context_window}, n_actions={self.n_actions}")
        print(f"Initial generated_frames shape: {self.generated_frames.shape}")
    
    def _to_vis(self, frames):
        """Convert frames to [0, 255] uint8 for display."""
        frames = frames.float().detach().cpu()
        
        # Handle [C, H, W] -> transpose to [H, W, C]
        if len(frames.shape) == 3:
            frames = frames.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # Take only first 3 channels if multi-channel
        if frames.shape[2] > 3:
            frames = frames[:, :, :3]
        elif frames.shape[2] == 1:
            frames = frames.squeeze(-1)  # Remove channel dimension for grayscale
        
        # Normalize to [0, 1]
        min_val = frames.min().item()
        max_val = frames.max().item()
        
        if min_val < 0:
            # Likely [-1, 1] range
            frames = (frames + 1.0) / 2.0
        elif max_val > 1:
            # Likely [0, 255] range
            frames = frames / 255.0
        
        frames = torch.clamp(frames, 0, 1)
        return (frames * 255).numpy().astype(np.uint8)
    
    def _get_action_latent(self, action_id: int):
        """
        Construct action_latent for the given action_id.
        Returns [1, context_window, action_dim] tensor.
        """
        # Take most recent actions from history
        recent_actions = self.action_history[-self.args.context_window:]
        
        # Pad with default action if needed
        if len(recent_actions) < self.args.context_window:
            pad_size = self.args.context_window - len(recent_actions)
            # Pad with zeros (neutral action)
            pad_actions = [0] * pad_size
            all_actions = pad_actions + recent_actions
        else:
            all_actions = recent_actions
        
        # Convert all actions to latents
        actions_tensor = torch.tensor(all_actions, device=self.device)
        action_latent = self.latent_action_model.quantizer.get_latents_from_indices(
            repeat(actions_tensor, 'i -> 1 i')
        )  # [1, context_window, action_dim]
        
        return action_latent
    
    def step(self, action_id: int):
        """
        Execute one inference step with the given action_id.
        """
        self.current_action_id = action_id
        self.action_history.append(action_id)
        
        with torch.inference_mode():
            # Tokenize context
            context_frames = self.generated_frames[:, -self.args.context_window:, :, :, :]
            video_indices = self.video_tokenizer.tokenize(context_frames)
            video_latents = self.video_tokenizer.quantizer.get_latents_from_indices(video_indices)
            
            # Get action latent
            action_latent = self._get_action_latent(action_id)
            
            # Dynamics inference
            def idx_to_latents(idx):
                return self.video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)
            
            next_video_latents = self.dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=self.args.prediction_horizon,
                num_steps=4,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=0.0,
            )
            
            # Detokenize and clamp
            next_frames = self.video_tokenizer.detokenize(next_video_latents)
            next_frames = next_frames.clamp(-1.0, 1.0)
            
            # Append to generated sequence (take last frame to keep sliding window)
            self.generated_frames = torch.cat(
                [self.generated_frames, next_frames[:, -self.args.prediction_horizon:, :, :, :]],
                dim=1
            )
            # Trim to keep memory bounded
            self.generated_frames = self.generated_frames[:, -self.args.context_window-5:, :, :, :]
        
        self.frame_count += 1
    
    def render(self):
        """Render current frame with overlay info."""
        # Get latest frame [C, H, W]
        latest_frame = self.generated_frames[0, -1, :, :, :]
        frame_img = self._to_vis(latest_frame)  # [H, W, C] or [H, W] uint8
        
        # Ensure 3-channel BGR format
        if len(frame_img.shape) == 2:
            # Grayscale to BGR
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        elif frame_img.shape[2] == 3:
            # RGB to BGR
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        else:
            # Should not happen with _to_vis, but guard anyway
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        display_size = 512
        h, w = frame_img.shape[:2]
        scale = min(display_size / h, display_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_img = cv2.resize(frame_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Add padding to make square
        canvas = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        y_offset = (display_size - new_h) // 2
        x_offset = (display_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_img
        
        # Calculate FPS
        current_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_tick - self.fps_clock + 1e-6)
        self.fps_clock = current_tick
        
        # Add overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(canvas, f"Action: {self.current_action_id}/{self.n_actions-1}", 
                   (10, 70), font, 1, (0, 255, 0), 2)
        cv2.putText(canvas, f"Frame: {self.frame_count}", (10, 110), font, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "ESC=Quit  WASD=Act  0-8=Direct", 
                   (10, display_size-20), font, 0.5, (200, 200, 200), 1)
        
        return canvas


def main():
    # Load config
    args: InferenceConfig = load_config(
        InferenceConfig,
        default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml')
    )
    
    # Override for interactive mode
    args.dataset = "POLE_POSITION"
    args.use_actions = True
    args.use_gt_actions = False
    args.use_interactive_mode = False  # We handle input manually
    args.prediction_horizon = 1
    args.context_window = 2
    
    # Auto-resolve checkpoints for Pole Position
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'results')
    
    # Find latest pole_position run
    pole_runs = sorted(
        [d for d in os.listdir(results_dir) if 'pole_position' in d.lower()],
        reverse=True
    )
    
    if pole_runs:
        latest_pole_run = os.path.join(results_dir, pole_runs[0])
        print(f"Using Pole Position run: {latest_pole_run}")
        
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer", run_root_dir=latest_pole_run)
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions", run_root_dir=latest_pole_run)
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics", run_root_dir=latest_pole_run)
    else:
        print("Warning: No Pole Position checkpoints found, using default search")
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer")
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions")
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics")
    
    print(f"Video tokenizer: {args.video_tokenizer_path}")
    print(f"Latent actions: {args.latent_actions_path}")
    print(f"Dynamics: {args.dynamics_path}")
    
    # Initialize game (Pole Position has 9 actions: 0-8)
    game = PolePositionGameState(args)
    
    # Action mapping
    action_map = {
        ord('w'): 1,      # up
        ord('a'): 3,      # left
        ord('s'): 2,      # down
        ord('d'): 0,      # right
        ord(' '): 0,      # space = neutral
    }
    # Direct numeric keys 0-8
    for i in range(9):
        action_map[ord('0') + i] = i
    
    print("\n=== Pole Position Interactive Game ===")
    print("Controls:")
    print("  W/A/S/D: action directions")
    print("  Space: neutral")
    print("  0-8: direct action index")
    print("  ESC: quit")
    print()
    
    # Game loop
    window_name = "Pole Position - Interactive Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 512)
    
    current_action = 0
    
    try:
        while True:
            # Render
            frame = game.render()
            cv2.imshow(window_name, frame)
            
            # Handle input (non-blocking, timeout=1ms)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Exiting...")
                break
            elif key in action_map:
                current_action = action_map[key]
            
            # Step inference
            game.step(current_action)
    
    finally:
        cv2.destroyAllWindows()
        print(f"Game ended. Total frames: {game.frame_count}")


if __name__ == "__main__":
    main()

