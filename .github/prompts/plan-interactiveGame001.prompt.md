## 任务：新增实时交互脚本 scripts/play_sonic.py（键盘控制 + OpenCV 实时显示）

### 目标
在 tinyworlds 现有推理链路基础上实现一个最小 Game Loop：
- 从 SONIC 数据集中随机取一个 clip 作为初始 context（2 帧）
- 每帧读取键盘输入映射到 action_id（0..n_actions-1）
- 走和 run_inference.py 完全一致的推理路径：
  context_frames -> video_tokenizer.tokenize -> get_latents_from_indices
  + action_latent -> dynamics_model.forward_inference -> video_tokenizer.detokenize
- 对 detokenize 输出做 clamp(-1,1)
- OpenCV 窗口实时渲染（ESC 退出），叠加 FPS 与 action_id

### 修改范围
只新增 `scripts/play_sonic.py`，不改模型代码，不改训练代码；复用：
- utils.config.load_config / InferenceConfig
- datasets.data_utils.load_data_and_data_loaders
- utils.inference_utils.load_models
- utils.utils.find_latest_checkpoint

### 关键实现点（必须）
1) 模型加载：与 scripts/run_inference.py 相同，用 use_latest_checkpoints 自动找 .pth
2) 初始 context：从 dataset 抽样，取 context_window=2 帧作为 generated_frames 起点
3) 每帧推理：
   - context_frames = generated_frames[:, -context_window:]
   - video_indices = video_tokenizer.tokenize(context_frames)
   - video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)
   - 构造 action_latent（shape 必须匹配 T = context_window + prediction_horizon）：
     - action_indices = tensor([[a0,a1,a2]], device)  # T=3 (2+1)
     - action_latent = latent_action_model.quantizer.get_latents_from_indices(action_indices)
   - next_latents = dynamics_model.forward_inference(... prediction_horizon=1, num_steps=1~4, temperature=0.0~0.5)
   - next_frame = video_tokenizer.detokenize(next_latents)
   - next_frame = next_frame.clamp(-1,1)
   - generated_frames = cat(generated_frames, next_frame[:, -1:])
4) 显示：[-1,1] -> [0,255]，RGB->BGR，resize 到 512，叠加 FPS/ACT
5) 交互：WASD/Space 映射 action_id（默认映射可先写死；并提供数字键 0-9 直接设 action_id 方便校准）

### 验收标准
- `python scripts/play_sonic.py` 会弹出窗口，持续更新画面
- 按 ESC 退出
- action_id 在屏幕上变化（按键有效）
- 无需 console 输入即可持续生成下一帧
