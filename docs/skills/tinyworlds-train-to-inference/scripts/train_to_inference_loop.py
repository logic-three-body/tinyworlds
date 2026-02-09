#!/usr/bin/env python3
"""Run TinyWorlds train->inference closed-loop verification."""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIMIZER_CHECKPOINT = "optim_state_dict.pt"
STATE_CHECKPOINT = "state.pt"

MODEL_ALIASES = {
    "video_tokenizer": ["video_tokenizer"],
    "latent_actions": ["latent_actions", "lam", "actions", "action_tokenizer"],
    "dynamics": ["dynamics"],
}

LEGACY_BASELINE = {
    "training.embed_dim": "(missing in training.yaml shared section)",
    "training.num_heads": "(missing in training.yaml shared section)",
    "training.hidden_dim": "(missing in training.yaml shared section)",
    "training.num_blocks": "(missing in training.yaml shared section)",
    "latent_actions.num_blocks": 2,
    "dynamics.learning_rate": 0.01,
    "dynamics.gradient_accumulation_steps": 4,
    "dynamics.log_interval": 2000,
    "dynamics.num_blocks": 8,
    "dynamics.video_tokenizer_path": "(hardcoded absolute path)",
    "dynamics.latent_actions_path": "(hardcoded absolute path)",
}


def parse_scalar(text):
    value = text.strip().split("#", 1)[0].strip()
    if not value:
        return None
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_bool(text):
    lower = str(text).strip().lower()
    if lower in ("1", "true", "yes", "y", "on"):
        return True
    if lower in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {text}")


def read_yaml_scalar(path, key, default=None):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        parsed = parse_scalar(match.group(1))
        return default if parsed is None else parsed
    return default


def load_alignment_snapshot(repo_root):
    training_yaml = Path(repo_root) / "configs/training.yaml"
    latent_yaml = Path(repo_root) / "configs/latent_actions.yaml"
    dynamics_yaml = Path(repo_root) / "configs/dynamics.yaml"
    return {
        "training.embed_dim": read_yaml_scalar(training_yaml, "embed_dim"),
        "training.num_heads": read_yaml_scalar(training_yaml, "num_heads"),
        "training.hidden_dim": read_yaml_scalar(training_yaml, "hidden_dim"),
        "training.num_blocks": read_yaml_scalar(training_yaml, "num_blocks"),
        "latent_actions.num_blocks": read_yaml_scalar(latent_yaml, "num_blocks"),
        "dynamics.learning_rate": read_yaml_scalar(dynamics_yaml, "learning_rate"),
        "dynamics.gradient_accumulation_steps": read_yaml_scalar(
            dynamics_yaml, "gradient_accumulation_steps"
        ),
        "dynamics.log_interval": read_yaml_scalar(dynamics_yaml, "log_interval"),
        "dynamics.num_blocks": read_yaml_scalar(dynamics_yaml, "num_blocks"),
        "dynamics.video_tokenizer_path": read_yaml_scalar(dynamics_yaml, "video_tokenizer_path"),
        "dynamics.latent_actions_path": read_yaml_scalar(dynamics_yaml, "latent_actions_path"),
    }


def step_from_name(name):
    match = re.search(r"_step_(\d+)$", name)
    return int(match.group(1)) if match else -1


def parse_last_json_line(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Failed to parse JSON output from subprocess.")


def run_process(cmd, cwd, env):
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    return proc


def validate_checkpoint_dir(path):
    if not path.is_dir():
        return False
    model = path / MODEL_CHECKPOINT
    optim = path / OPTIMIZER_CHECKPOINT
    state = path / STATE_CHECKPOINT
    if not (model.is_file() and optim.is_file() and state.is_file()):
        return False
    if model.stat().st_size < 1000 or optim.stat().st_size < 1000:
        return False
    return True


def resolve_latest_valid_checkpoint(path_str):
    if not path_str:
        raise ValueError("Empty checkpoint path.")
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if validate_checkpoint_dir(path):
        return str(path)
    candidates = [
        p for p in path.glob("*_step_*") if p.is_dir() and validate_checkpoint_dir(p)
    ]
    if not candidates:
        raise RuntimeError(f"No valid checkpoint found under: {path}")
    candidates.sort(key=lambda p: (step_from_name(p.name), p.stat().st_mtime))
    return str(candidates[-1].resolve())


def checkpoint_to_model_file(path_str):
    path = Path(path_str).resolve()
    if path.is_file():
        return str(path)
    model_file = path / MODEL_CHECKPOINT
    if model_file.is_file():
        return str(model_file.resolve())
    raise FileNotFoundError(
        f"Checkpoint model file not found. Expected file path or directory containing "
        f"'{MODEL_CHECKPOINT}': {path}"
    )


def find_latest_model_checkpoint(repo_root, model_name, prefer_root=None):
    aliases = MODEL_ALIASES.get(model_name, [model_name])
    search_roots = []
    if prefer_root:
        search_roots.append(Path(prefer_root))
    search_roots.append(Path(repo_root) / "results")

    for root in search_roots:
        if not root.exists():
            continue
        candidates = []
        for alias in aliases:
            for match in root.glob(f"**/*{alias}_step_*"):
                if match.is_dir() and validate_checkpoint_dir(match):
                    candidates.append(match.resolve())
        if candidates:
            candidates.sort(key=lambda p: (p.stat().st_mtime, step_from_name(p.name)))
            return str(candidates[-1])
    return None


def list_inference_pngs(repo_root):
    png_dir = Path(repo_root) / "inference_results"
    if not png_dir.exists():
        return []
    files = [p for p in png_dir.glob("inference_results_gt_vs_pred*.png") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def run_auto_train_loop(
    args,
    repo_root,
    env,
    only_stage,
    run_root,
    video_checkpoint=None,
    latent_checkpoint=None,
    dynamics_checkpoint=None,
    target_updates=0,
    init_learning_rate=None,
):
    script = Path(repo_root) / "docs/skills/tinyworlds-training-causal-diagnostics/scripts/auto_train_loop.py"
    cmd = [
        args.python_exec,
        str(script),
        "--repo-root",
        repo_root,
        "--python-exec",
        args.python_exec,
        "--torchrun",
        args.torchrun,
        "--training-config",
        args.training_config,
        "--video-config",
        args.video_config,
        "--latent-config",
        args.latent_config,
        "--dynamics-config",
        args.dynamics_config,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--max-retries",
        str(args.max_train_retries),
        "--cleanup-extra-processes",
        str(args.cleanup_extra_processes).lower(),
        "--enforce-dual-gpu-check",
        str(args.enforce_dual_gpu_check).lower(),
        "--monitor-interval-sec",
        str(args.monitor_interval_sec),
        "--gpu-util-threshold",
        str(args.gpu_util_threshold),
        "--gpu-required-samples",
        str(args.gpu_required_samples),
        "--video-chunk",
        str(args.video_chunk),
        "--latent-chunk",
        str(args.latent_chunk),
        "--dynamics-chunk",
        str(args.dynamics_chunk),
        "--run-root",
        run_root,
        "--only-stage",
        only_stage,
    ]
    if target_updates > 0:
        cmd += ["--target-updates", str(target_updates)]
    if init_learning_rate is not None:
        cmd += ["--init-learning-rate", str(init_learning_rate)]
    if args.init_grad_accum is not None:
        cmd += ["--init-grad-accum", str(args.init_grad_accum)]
    if args.init_batch_size is not None:
        cmd += ["--init-batch-size", str(args.init_batch_size)]
    if args.init_log_interval is not None:
        cmd += ["--init-log-interval", str(args.init_log_interval)]

    if video_checkpoint:
        cmd += ["--video-checkpoint", video_checkpoint]
    if latent_checkpoint:
        cmd += ["--latent-checkpoint", latent_checkpoint]
    if dynamics_checkpoint:
        cmd += ["--dynamics-checkpoint", dynamics_checkpoint]
    if only_stage == "dynamics":
        if not video_checkpoint or not latent_checkpoint:
            raise ValueError("dynamics training requires both video and latent checkpoints")

    proc = run_process(cmd, cwd=repo_root, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"auto_train_loop failed with exit code {proc.returncode}")
    return parse_last_json_line(proc.stdout)


def run_standard_inference_gate(args, repo_root, env, video_ckpt, latent_ckpt, dynamics_ckpt):
    video_checkpoint_dir = resolve_latest_valid_checkpoint(video_ckpt)
    latent_checkpoint_dir = resolve_latest_valid_checkpoint(latent_ckpt)
    dynamics_checkpoint_dir = resolve_latest_valid_checkpoint(dynamics_ckpt)

    before = {str(p.resolve()) for p in list_inference_pngs(repo_root)}
    started_at = time.time()
    cmd = [
        args.python_exec,
        "scripts/run_inference.py",
        "--config",
        args.inference_config,
        "--",
        "use_latest_checkpoints=false",
        f"dataset={args.dataset}",
        f"device={args.device}",
        "use_actions=true",
        "use_gt_actions=false",
        "use_interactive_mode=false",
        f"teacher_forced={str(args.teacher_forced).lower()}",
        f"context_window={args.context_window}",
        f"prediction_horizon={args.prediction_horizon}",
        f"generation_steps={args.generation_steps}",
        f"preload_ratio={args.preload_ratio}",
        f"temperature={args.temperature}",
        f"video_tokenizer_path={video_checkpoint_dir}",
        f"latent_actions_path={latent_checkpoint_dir}",
        f"dynamics_path={dynamics_checkpoint_dir}",
    ]
    proc = run_process(cmd, cwd=repo_root, env=env)
    output = f"{proc.stdout}\n{proc.stderr}".strip()

    markers = {
        "video_checkpoint_printed": "Using video_tokenizer checkpoint:" in output,
        "latent_checkpoint_printed": "Using latent_actions checkpoint:" in output,
        "dynamics_checkpoint_printed": "Using dynamics checkpoint:" in output,
        "inference_loop_ran": "Inferring frame " in output,
        "inference_stats_printed": "Inference stats:" in output,
    }
    mse_match = re.search(r"Mean Squared Error \(GT vs Pred\):\s*([0-9.]+)", output)
    mse = float(mse_match.group(1)) if mse_match else None

    new_png = None
    for candidate in list_inference_pngs(repo_root):
        resolved = str(candidate.resolve())
        if resolved in before:
            continue
        if candidate.stat().st_mtime >= started_at - 2:
            new_png = resolved
    mse_ok = mse is not None and (args.max_mse <= 0 or mse <= args.max_mse)
    pass_ok = (
        proc.returncode == 0
        and all(markers.values())
        and new_png is not None
        and mse_ok
    )

    return {
        "returncode": proc.returncode,
        "markers": markers,
        "mse": mse,
        "mse_threshold": args.max_mse,
        "mse_ok": mse_ok,
        "new_png": new_png,
        "passed": pass_ok,
        "selected_checkpoints": {
            "video_tokenizer": video_checkpoint_dir,
            "latent_actions": latent_checkpoint_dir,
            "dynamics": dynamics_checkpoint_dir,
        },
    }


def run_git_audit(args, repo_root, env, timestamp):
    script = Path(repo_root) / "docs/skills/tinyworlds-train-to-inference/scripts/git_zelda_audit.py"
    cmd = [
        args.python_exec,
        str(script),
        "--repo-root",
        repo_root,
        "--year",
        str(args.history_audit_year),
        "--mode",
        args.history_audit_mode,
        "--out-dir",
        args.analysis_out_dir,
        "--timestamp",
        timestamp,
    ]
    proc = run_process(cmd, cwd=repo_root, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git_zelda_audit failed with exit code {proc.returncode}")
    return parse_last_json_line(proc.stdout)


def run_wandb_analysis(args, repo_root, env, timestamp, run_root, stage_run_ids):
    script = Path(repo_root) / "docs/skills/tinyworlds-train-to-inference/scripts/wandb_stage_analyzer.py"
    project = args.wandb_project or read_yaml_scalar(Path(repo_root) / args.training_config, "wandb_project", "tinyworlds")
    cmd = [
        args.python_exec,
        str(script),
        "--repo-root",
        repo_root,
        "--run-root",
        run_root,
        "--wandb-project",
        str(project),
        "--wandb-source",
        args.wandb_source,
        "--analysis-out-dir",
        args.analysis_out_dir,
        "--timestamp",
        timestamp,
    ]
    for stage, run_id in sorted((stage_run_ids or {}).items()):
        if run_id:
            cmd += ["--stage-run-id", f"{stage}={run_id}"]

    proc = run_process(cmd, cwd=repo_root, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"wandb_stage_analyzer failed with exit code {proc.returncode}")
    return parse_last_json_line(proc.stdout)


def write_report(report_path, report):
    lines = [
        f"# TinyWorlds Train-to-Inference Report ({report['timestamp']})",
        "",
        f"- status: `{report['status']}`",
        f"- repo_root: `{report['repo_root']}`",
        f"- run_root: `{report['run_root']}`",
        f"- train_stage: `{report['train_stage']}`",
        f"- checkpoint_policy: `{report['checkpoint_policy']}`",
        f"- teacher_forced: `{report.get('teacher_forced')}`",
        f"- cleanup_extra_processes: `{report.get('cleanup_extra_processes')}`",
        f"- enforce_dual_gpu_check: `{report.get('enforce_dual_gpu_check')}`",
        f"- monitor_interval_sec: `{report.get('monitor_interval_sec')}`",
        f"- gpu_util_threshold: `{report.get('gpu_util_threshold')}`",
        f"- gpu_required_samples: `{report.get('gpu_required_samples')}`",
        "",
        "## Checkpoints",
        f"- video_tokenizer: `{report['checkpoints']['video_tokenizer']}`",
        f"- latent_actions: `{report['checkpoints']['latent_actions']}`",
        f"- dynamics: `{report['checkpoints']['dynamics']}`",
        "",
        "## Stage Run IDs",
    ]
    for stage in ["video_tokenizer", "latent_actions", "dynamics"]:
        lines.append(f"- {stage}: `{report['stage_run_ids'].get(stage)}`")

    lines.append("")
    lines.append("## Inference Attempts")
    for idx, attempt in enumerate(report["inference_attempts"], start=1):
        lines.append(f"### Attempt {idx}")
        lines.append(f"- passed: `{attempt['passed']}`")
        lines.append(f"- mse: `{attempt['mse']}`")
        lines.append(f"- mse_threshold: `{attempt['mse_threshold']}`")
        lines.append(f"- new_png: `{attempt['new_png']}`")
        lines.append(f"- returncode: `{attempt['returncode']}`")
        lines.append("")

    lines.append("## Training Calls")
    for item in report["training_calls"]:
        lines.append(f"- stage `{item['stage']}` target_updates `{item.get('target_updates')}`")
        lines.append(f"  - deps: `{item.get('deps')}`")
        lines.append(f"  - report: `{item.get('report')}`")
        lines.append(f"  - stage_run_ids: `{item.get('stage_run_ids')}`")

    lines += ["", "## History Audit"]
    hist = report.get("history_audit") or {}
    lines.append(f"- report: `{hist.get('report')}`")
    lines.append(f"- year: `{hist.get('year')}`")
    lines.append(f"- mode: `{hist.get('mode')}`")
    lines.append(f"- total_commits_in_year: `{hist.get('total_commits_in_year')}`")
    lines.append(f"- selected_count: `{hist.get('selected_count')}`")
    lines.append(f"- relevant_file_touches: `{hist.get('relevant_file_touches')}`")

    lines += ["", "## W&B Analysis"]
    wb = report.get("wandb_analysis") or {}
    lines.append(f"- enabled: `{report.get('enable_wandb_analysis')}`")
    lines.append(f"- source_mode: `{report.get('wandb_source')}`")
    lines.append(f"- report: `{wb.get('report')}`")
    lines.append(f"- json: `{wb.get('json')}`")
    lines.append(f"- plot_dir: `{wb.get('plot_dir')}`")

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def write_retrospective(retro_path, report, alignment_snapshot):
    best_attempt = None
    for attempt in report["inference_attempts"]:
        if attempt.get("passed"):
            best_attempt = attempt
            break
    if best_attempt is None and report["inference_attempts"]:
        best_attempt = report["inference_attempts"][-1]

    hist = report.get("history_audit") or {}
    wb = report.get("wandb_analysis") or {}
    lines = [
        f"# ZELDA Train-to-Inference Retrospective ({report['timestamp']})",
        "",
        "## 1) 闭环结果",
        f"- 状态: `{report['status']}`",
        f"- run_root: `{report['run_root']}`",
        f"- inference_teacher_forced: `{report.get('teacher_forced')}`",
        f"- final_video_checkpoint: `{report['checkpoints'].get('video_tokenizer')}`",
        f"- final_latent_checkpoint: `{report['checkpoints'].get('latent_actions')}`",
        f"- final_dynamics_checkpoint: `{report['checkpoints'].get('dynamics')}`",
    ]
    if best_attempt is not None:
        lines += [
            f"- inference_mse: `{best_attempt.get('mse')}`",
            f"- mse_threshold: `{best_attempt.get('mse_threshold')}`",
            f"- inference_png: `{best_attempt.get('new_png')}`",
        ]

    lines += [
        "",
        "## 2) 2025 历史证据",
        f"- audit_report: `{hist.get('report')}`",
        f"- year: `{hist.get('year')}`",
        f"- mode: `{hist.get('mode')}`",
        f"- total_commits_in_year: `{hist.get('total_commits_in_year')}`",
        f"- strict_selected_count: `{hist.get('selected_count')}`",
        f"- relevant_file_touches: `{hist.get('relevant_file_touches')}`",
        f"- selected_commits: `{hist.get('selected_commits')}`",
        "",
        "## 3) 旧参数版本不达标的核心原因",
        "| 项目 | 旧值 | 新值 | 影响 |",
        "|---|---|---|---|",
    ]

    for key in [
        "training.embed_dim",
        "training.num_heads",
        "training.hidden_dim",
        "training.num_blocks",
        "latent_actions.num_blocks",
        "dynamics.num_blocks",
        "dynamics.learning_rate",
        "dynamics.gradient_accumulation_steps",
        "dynamics.log_interval",
        "dynamics.video_tokenizer_path",
        "dynamics.latent_actions_path",
    ]:
        old_value = LEGACY_BASELINE.get(key)
        new_value = alignment_snapshot.get(key)
        if key.startswith("training.") and "missing" in str(old_value):
            impact = "共享骨干参数未集中管理，三阶段容易隐式漂移。"
        elif key == "latent_actions.num_blocks":
            impact = "latent 容量低于 video/dynamics，动作语义对齐不足。"
        elif key == "dynamics.num_blocks":
            impact = "dynamics 结构与共享骨干不一致，跨模块表征映射不稳定。"
        elif key == "dynamics.learning_rate":
            impact = "学习率过高导致 dynamics 收敛抖动，推理漂移风险增大。"
        elif key == "dynamics.gradient_accumulation_steps":
            impact = "梯度累积策略与当前 batch 目标不匹配，更新噪声偏大。"
        elif key == "dynamics.log_interval":
            impact = "日志/可视化反馈过稀，门禁回路纠错滞后。"
        else:
            impact = "硬编码路径易引用到历史 run，训练-推理 checkpoint 不一致。"
        lines.append(f"| `{key}` | `{old_value}` | `{new_value}` | {impact} |")

    lines += [
        "",
        "## 4) 新参数配置为何成功",
        "- 通过 `configs/training.yaml` 统一 shared backbone 参数（`embed_dim/num_heads/hidden_dim/num_blocks`），避免 video-latent-dynamics 结构漂移。",
        f"- 将 dynamics 的学习率从 `{LEGACY_BASELINE['dynamics.learning_rate']}` 降到 `{alignment_snapshot.get('dynamics.learning_rate')}`，降低震荡并提高收敛稳定性。",
        "- 将 `dynamics` 的 checkpoint 路径改为运行时注入（配置中 `null`），避免跨 run 误用旧权重。",
        "- 训练后立即运行标准推理门禁，用同一组 checkpoint 验证端到端行为，确保“能训完”且“能推理”。",
    ]
    if report.get("teacher_forced"):
        lines += [
            "",
            "## 4.1) 推理模式说明",
            "- 本次通过样例使用 `teacher_forced=true`。",
            "- 该结果用于验证闭环链路可执行与参数对齐，不等价于严格自回归(`teacher_forced=false`)达标。",
        ]

    lines += [
        "",
        "## 5) W&B分析与产物",
        f"- wandb_analysis_report: `{wb.get('report')}`",
        f"- wandb_analysis_json: `{wb.get('json')}`",
        f"- wandb_plots_dir: `{wb.get('plot_dir')}`",
        "",
        "## 6) 后续技能流程固化",
        "- 每次闭环通过后，必须在 `docs/action/` 生成反思报告、W&B分析报告、2025历史审计报告。",
        "- 闭环完成条件不再只看训练和推理，还必须包含三类文档产物。",
    ]
    Path(retro_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="TinyWorlds train->inference closed-loop validator.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-exec", default="./.venv/bin/python")
    parser.add_argument("--torchrun", default="./.venv/bin/torchrun")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--video-config", default="configs/video_tokenizer.yaml")
    parser.add_argument("--latent-config", default="configs/latent_actions.yaml")
    parser.add_argument("--dynamics-config", default="configs/dynamics.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--run-root", default="")

    parser.add_argument(
        "--train-stage",
        choices=["none", "all", "video_tokenizer", "latent_actions", "dynamics"],
        default="all",
    )
    parser.add_argument("--checkpoint-policy", choices=["fresh", "latest"], default="fresh")
    parser.add_argument("--video-checkpoint", default="")
    parser.add_argument("--latent-checkpoint", default="")
    parser.add_argument("--dynamics-checkpoint", default="")
    parser.add_argument("--target-updates", type=int, default=0)
    parser.add_argument("--max-train-retries", type=int, default=5)
    parser.add_argument("--cleanup-extra-processes", type=parse_bool, default=True)
    parser.add_argument("--enforce-dual-gpu-check", type=parse_bool, default=True)
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)
    parser.add_argument("--gpu-util-threshold", type=float, default=1.0)
    parser.add_argument("--gpu-required-samples", type=int, default=2)
    parser.add_argument("--video-chunk", type=int, default=5000)
    parser.add_argument("--latent-chunk", type=int, default=1000)
    parser.add_argument("--dynamics-chunk", type=int, default=2000)
    parser.add_argument("--init-learning-rate", type=float, default=None)
    parser.add_argument("--init-grad-accum", type=int, default=None)
    parser.add_argument("--init-batch-size", type=int, default=None)
    parser.add_argument("--init-log-interval", type=int, default=None)

    parser.add_argument("--dataset", default="ZELDA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-steps", type=int, default=12)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--preload-ratio", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--max-mse", type=float, default=0.03)

    parser.add_argument("--max-inference-retries", type=int, default=2)
    parser.add_argument("--retry-chunk", type=int, default=2000)
    parser.add_argument("--retry-init-learning-rate", type=float, default=0.0005)
    parser.add_argument("--retry-lr-decay", type=float, default=0.7)
    parser.add_argument("--retry-lr-floor", type=float, default=1e-6)

    parser.add_argument("--enable-wandb-analysis", type=parse_bool, default=True)
    parser.add_argument(
        "--wandb-source",
        choices=["api_first", "api_only", "local_only"],
        default="api_first",
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--analysis-out-dir", default="docs/action")

    parser.add_argument("--history-audit-year", type=int, default=2025)
    parser.add_argument(
        "--history-audit-mode",
        choices=["strict", "wide", "current_paths"],
        default="strict",
    )
    return parser.parse_args()


def initialize_checkpoints(args, repo_root, run_root):
    if args.checkpoint_policy == "latest":
        video_ckpt = args.video_checkpoint or find_latest_model_checkpoint(repo_root, "video_tokenizer", run_root)
        latent_ckpt = args.latent_checkpoint or find_latest_model_checkpoint(repo_root, "latent_actions", run_root)
        dynamics_ckpt = args.dynamics_checkpoint or find_latest_model_checkpoint(repo_root, "dynamics", run_root)
    else:
        video_ckpt = args.video_checkpoint
        latent_ckpt = args.latent_checkpoint
        dynamics_ckpt = args.dynamics_checkpoint
    return video_ckpt, latent_ckpt, dynamics_ckpt


def main():
    args = parse_args()
    repo_root = str(Path(args.repo_root).resolve())
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    run_root = args.run_root or str(Path(repo_root) / "results" / f"train_to_inference_{timestamp}")
    os.makedirs(run_root, exist_ok=True)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{old_pythonpath}" if old_pythonpath else repo_root

    report = {
        "timestamp": timestamp,
        "repo_root": repo_root,
        "run_root": run_root,
        "train_stage": args.train_stage,
        "checkpoint_policy": args.checkpoint_policy,
        "teacher_forced": bool(args.teacher_forced),
        "training_calls": [],
        "inference_attempts": [],
        "checkpoints": {},
        "status": "failed",
        "stage_run_ids": {},
        "history_audit": {},
        "wandb_analysis": {},
        "enable_wandb_analysis": args.enable_wandb_analysis,
        "wandb_source": args.wandb_source,
        "cleanup_extra_processes": args.cleanup_extra_processes,
        "enforce_dual_gpu_check": args.enforce_dual_gpu_check,
        "monitor_interval_sec": args.monitor_interval_sec,
        "gpu_util_threshold": args.gpu_util_threshold,
        "gpu_required_samples": args.gpu_required_samples,
    }

    report["history_audit"] = run_git_audit(args, repo_root, env, timestamp)

    video_ckpt, latent_ckpt, dynamics_ckpt = initialize_checkpoints(args, repo_root, run_root)

    if args.train_stage != "none":
        train_out = run_auto_train_loop(
            args=args,
            repo_root=repo_root,
            env=env,
            only_stage=args.train_stage,
            run_root=run_root,
            video_checkpoint=video_ckpt,
            latent_checkpoint=latent_ckpt,
            dynamics_checkpoint=dynamics_ckpt,
            target_updates=args.target_updates,
            init_learning_rate=args.init_learning_rate,
        )
        stage_run_ids = train_out.get("stage_run_ids", {}) or {}
        report["training_calls"].append(
            {
                "stage": args.train_stage,
                "target_updates": args.target_updates,
                "deps": train_out.get("deps"),
                "report": train_out.get("report"),
                "stage_run_ids": stage_run_ids,
            }
        )
        deps = train_out.get("deps", {}) or {}
        video_ckpt = deps.get("video_tokenizer", video_ckpt)
        latent_ckpt = deps.get("latent_actions", latent_ckpt)
        dynamics_ckpt = deps.get("dynamics", dynamics_ckpt)
        report["stage_run_ids"].update(stage_run_ids)

    if args.checkpoint_policy == "latest":
        if not video_ckpt:
            video_ckpt = find_latest_model_checkpoint(repo_root, "video_tokenizer")
        if not latent_ckpt:
            latent_ckpt = find_latest_model_checkpoint(repo_root, "latent_actions")
        if not dynamics_ckpt:
            dynamics_ckpt = find_latest_model_checkpoint(repo_root, "dynamics")

    if not video_ckpt or not latent_ckpt or not dynamics_ckpt:
        raise RuntimeError(
            "Missing checkpoints after training. With checkpoint-policy=fresh, pass --train-stage all or explicit --*-checkpoint paths."
        )

    video_ckpt = resolve_latest_valid_checkpoint(video_ckpt)
    latent_ckpt = resolve_latest_valid_checkpoint(latent_ckpt)
    dynamics_ckpt = resolve_latest_valid_checkpoint(dynamics_ckpt)

    retry_lr = args.retry_init_learning_rate
    for attempt in range(args.max_inference_retries + 1):
        inf = run_standard_inference_gate(
            args=args,
            repo_root=repo_root,
            env=env,
            video_ckpt=video_ckpt,
            latent_ckpt=latent_ckpt,
            dynamics_ckpt=dynamics_ckpt,
        )
        report["inference_attempts"].append(inf)
        if inf["passed"]:
            report["status"] = "passed"
            break
        if attempt >= args.max_inference_retries:
            break

        current_step = step_from_name(Path(dynamics_ckpt).name)
        retry_target = current_step + args.retry_chunk if current_step >= 0 else args.retry_chunk
        train_out = run_auto_train_loop(
            args=args,
            repo_root=repo_root,
            env=env,
            only_stage="dynamics",
            run_root=run_root,
            video_checkpoint=video_ckpt,
            latent_checkpoint=latent_ckpt,
            dynamics_checkpoint=dynamics_ckpt,
            target_updates=retry_target,
            init_learning_rate=retry_lr,
        )
        stage_run_ids = train_out.get("stage_run_ids", {}) or {}
        report["training_calls"].append(
            {
                "stage": "dynamics",
                "target_updates": retry_target,
                "deps": train_out.get("deps"),
                "report": train_out.get("report"),
                "retry_learning_rate": retry_lr,
                "stage_run_ids": stage_run_ids,
            }
        )
        deps = train_out.get("deps", {}) or {}
        dynamics_ckpt = resolve_latest_valid_checkpoint(deps.get("dynamics", dynamics_ckpt))
        report["stage_run_ids"].update(stage_run_ids)
        retry_lr = max(retry_lr * args.retry_lr_decay, args.retry_lr_floor)

    report["checkpoints"] = {
        "video_tokenizer": video_ckpt,
        "latent_actions": latent_ckpt,
        "dynamics": dynamics_ckpt,
    }

    if args.enable_wandb_analysis:
        report["wandb_analysis"] = run_wandb_analysis(
            args=args,
            repo_root=repo_root,
            env=env,
            timestamp=timestamp,
            run_root=run_root,
            stage_run_ids=report["stage_run_ids"],
        )

    report_path = Path(repo_root) / "docs/action" / f"train-to-inference-report-{timestamp}.md"
    write_report(report_path, report)
    retro_path = Path(repo_root) / "docs/action" / f"train-to-inference-retrospective-{timestamp}.md"
    write_retrospective(retro_path, report, load_alignment_snapshot(repo_root))

    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "retrospective": str(retro_path),
                "history_audit": report.get("history_audit", {}).get("report"),
                "wandb_analysis": report.get("wandb_analysis", {}).get("report"),
                "checkpoints": report["checkpoints"],
                "stage_run_ids": report["stage_run_ids"],
                "inference_attempts": len(report["inference_attempts"]),
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
