# TinyWorlds Auto Training Loop Report (2026_02_08_18_55_00)

- run_root: `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00`
- nproc_per_node: `2`

## video_tokenizer
- target_updates: `120`
- chunk_size: `120`
- initial_checkpoint: `/root/tinyworlds/results/2026_02_04_19_51_20/video_tokenizer/checkpoints/video_tokenizer_step_37500`
- final_checkpoint: `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/video_tokenizer/checkpoints/video_tokenizer_step_100`
- gates:
  - gate `120` decision `continue` checkpoint `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/video_tokenizer/checkpoints/video_tokenizer_step_100`

## latent_actions
- target_updates: `120`
- chunk_size: `120`
- initial_checkpoint: `/root/tinyworlds/results/2026_02_05_21_05_51/latent_actions/checkpoints/latent_actions_step_9500`
- final_checkpoint: `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/latent_actions/checkpoints/latent_actions_step_100`
- gates:
  - gate `120` decision `continue` checkpoint `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/latent_actions/checkpoints/latent_actions_step_100`

## dynamics
- target_updates: `120`
- chunk_size: `120`
- initial_checkpoint: `/root/tinyworlds/results/auto_train_loop_2026_02_07_23_14_07/dynamics/checkpoints/dynamics_step_106000`
- final_checkpoint: `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/dynamics/checkpoints/dynamics_step_106120`
- gates:
  - gate `120` decision `continue` checkpoint `/root/tinyworlds/results/train_to_inference_2026_02_08_18_55_00/dynamics/checkpoints/dynamics_step_106120`
