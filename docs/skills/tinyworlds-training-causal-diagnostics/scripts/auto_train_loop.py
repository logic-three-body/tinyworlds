#!/usr/bin/env python3
"""Automate TinyWorlds gated training with retune/retry loops."""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def parse_scalar(text):
    s = text.strip().split("#", 1)[0].strip()
    if not s:
        return None
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def read_yaml_scalar(path, key, default=None):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            v = parse_scalar(m.group(1))
            return default if v is None else v
    return default


def run_cmd(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def step_from_name(name):
    m = re.search(r"_step_(\d+)$", name)
    return int(m.group(1)) if m else -1


def validate_checkpoint_dir(path):
    model = path / "model_state_dict.pt"
    optim = path / "optim_state_dict.pt"
    state = path / "state.pt"
    if not (model.exists() and optim.exists() and state.exists()):
        return False
    if model.stat().st_size < 1000 or optim.stat().st_size < 1000:
        return False
    try:
        import torch  # pylint: disable=import-outside-toplevel

        md = torch.load(model, map_location="cpu", weights_only=False)
        od = torch.load(optim, map_location="cpu", weights_only=False)
        if not isinstance(md, dict) or not md:
            return False
        if not isinstance(od, dict) or not od:
            return False
    except Exception:
        return False
    return True


def latest_checkpoints(stage_checkpoints_dir):
    ckpts = [p for p in Path(stage_checkpoints_dir).glob("*_step_*") if p.is_dir()]
    if not ckpts:
        return None, None
    ckpts.sort(key=lambda p: step_from_name(p.name))
    latest = ckpts[-1]
    valid = [p for p in ckpts if validate_checkpoint_dir(p)]
    latest_valid = valid[-1] if valid else None
    return latest, latest_valid


def resolve_latest_valid_checkpoint(path_str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_str}")
    if validate_checkpoint_dir(p):
        return str(p.resolve())
    latest_any, latest_valid = latest_checkpoints(p)
    if latest_valid is None:
        raise RuntimeError(
            f"No valid checkpoint found under {path_str}; latest candidate: {latest_any}"
        )
    return str(latest_valid.resolve())


def find_new_wandb_run(root, since_ts):
    wandb_dir = Path(root) / "wandb"
    runs = [p for p in wandb_dir.glob("run-*") if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime)
    recent = [r for r in runs if r.stat().st_mtime >= since_ts - 2]
    return (recent[-1] if recent else runs[-1]).resolve()


def export_stage_history(python_exec, repo_root, stage, run_dir, suffix):
    run_name = run_dir.name
    run_id = run_name.split("-")[-1]
    run_file = run_dir / f"run-{run_id}.wandb"
    exporter = (
        Path(repo_root)
        / "docs/skills/tinyworlds-training-diagnostics/scripts/export_local_wandb_history.py"
    )
    out_dir = Path(repo_root) / "Logs"
    run_cmd(
        [
            python_exec,
            str(exporter),
            "--run",
            f"{stage}={run_file}",
            "--out-dir",
            str(out_dir),
            "--suffix",
            suffix,
        ]
    )
    csvs = sorted(
        out_dir.glob(f"{stage}_history_*_{suffix}.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return csvs[-1] if csvs else None


def gate_verdict(python_exec, repo_root, stage, csv_path, recent_rows):
    scorer = (
        Path(repo_root)
        / "docs/skills/tinyworlds-training-causal-diagnostics/scripts/stage_gate_verdict.py"
    )
    cmd = [python_exec, str(scorer), "--stage", stage, "--csv", str(csv_path)]
    if recent_rows > 0:
        cmd += ["--recent-rows", str(recent_rows)]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def latest_visualization(stage, run_root):
    vis_dir = Path(run_root) / stage / "visualizations"
    if not vis_dir.exists():
        return None
    files = [p for p in vis_dir.glob("*.png") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def stage_script(stage):
    mapping = {
        "video_tokenizer": "scripts/train_video_tokenizer.py",
        "latent_actions": "scripts/train_latent_actions.py",
        "dynamics": "scripts/train_dynamics.py",
    }
    return mapping[stage]


def stage_config(stage, args):
    mapping = {
        "video_tokenizer": args.video_config,
        "latent_actions": args.latent_config,
        "dynamics": args.dynamics_config,
    }
    return mapping[stage]


def retune(stage, tune, reason="gate_fail"):
    if reason == "runtime_error":
        # First response to runtime failures (e.g. OOM): reduce per-GPU batch.
        if tune["batch_size_per_gpu"] > 1:
            tune["batch_size_per_gpu"] = max(1, tune["batch_size_per_gpu"] // 2)
            return tune
    if stage == "video_tokenizer":
        tune["learning_rate"] = max(tune["learning_rate"] * 0.5, 1e-6)
        tune["gradient_accumulation_steps"] = max(1, tune["gradient_accumulation_steps"] + 1)
        tune["log_interval"] = max(500, tune["log_interval"] // 2)
    elif stage == "latent_actions":
        tune["learning_rate"] = max(tune["learning_rate"] * 0.8, 1e-6)
        tune["log_interval"] = max(250, tune["log_interval"] // 2)
    elif stage == "dynamics":
        tune["learning_rate"] = max(tune["learning_rate"] * 0.2, 1e-6)
        tune["gradient_accumulation_steps"] = max(1, tune["gradient_accumulation_steps"] - 1)
        tune["log_interval"] = max(500, tune["log_interval"] // 2)
    return tune


def run_stage(args, stage, deps, report, initial_checkpoint=None):
    cfg_path = Path(args.repo_root) / stage_config(stage, args)
    target = int(read_yaml_scalar(cfg_path, "n_updates", 10000))
    if args.target_updates > 0:
        target = int(args.target_updates)
    tune = {
        "batch_size_per_gpu": int(read_yaml_scalar(cfg_path, "batch_size_per_gpu", 8)),
        "learning_rate": float(read_yaml_scalar(cfg_path, "learning_rate", 1e-4)),
        "gradient_accumulation_steps": int(
            read_yaml_scalar(cfg_path, "gradient_accumulation_steps", 1)
        ),
        "log_interval": int(read_yaml_scalar(cfg_path, "log_interval", 1000)),
    }
    if args.init_batch_size is not None:
        tune["batch_size_per_gpu"] = int(args.init_batch_size)
    if args.init_learning_rate is not None:
        tune["learning_rate"] = float(args.init_learning_rate)
    if args.init_grad_accum is not None:
        tune["gradient_accumulation_steps"] = int(args.init_grad_accum)
    if args.init_log_interval is not None:
        tune["log_interval"] = int(args.init_log_interval)
    chunk = int(args.chunk_size.get(stage, 1000))
    completed = 0
    retries = 0
    accepted_ckpt = None
    if initial_checkpoint:
        accepted_ckpt = resolve_latest_valid_checkpoint(initial_checkpoint)
    initial_ckpt_resolved = accepted_ckpt
    gates = []

    while completed < target:
        gate_updates = min(chunk, target - completed)
        start_step = step_from_name(Path(accepted_ckpt).name) if accepted_ckpt else -1
        run_target = (start_step + 1 + gate_updates) if accepted_ckpt else gate_updates
        vis_before = latest_visualization(stage, args.run_root)
        vis_before_mtime = vis_before.stat().st_mtime if vis_before else -1.0
        cmd = [
            args.torchrun,
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            stage_script(stage),
            "--config",
            stage_config(stage, args),
            "--training_config",
            args.training_config,
            f"n_updates={run_target}",
            f"batch_size_per_gpu={tune['batch_size_per_gpu']}",
            f"learning_rate={tune['learning_rate']}",
            f"gradient_accumulation_steps={tune['gradient_accumulation_steps']}",
            f"log_interval={tune['log_interval']}",
        ]
        if accepted_ckpt:
            cmd.append(f"checkpoint={accepted_ckpt}")
        if stage == "dynamics":
            cmd.append(f"video_tokenizer_path={deps['video_tokenizer']}")
            cmd.append(f"latent_actions_path={deps['latent_actions']}")

        start = time.time()
        env = os.environ.copy()
        env["NG_RUN_ROOT_DIR"] = args.run_root
        repo_abs = str(Path(args.repo_root).resolve())
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_abs}:{old_pp}" if old_pp else repo_abs
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            run_cmd(cmd, env=env)
        except subprocess.CalledProcessError as exc:
            retries += 1
            gates.append(
                {
                    "gate": completed + gate_updates,
                    "decision": "retune",
                    "reason": f"runtime_error_exit_{exc.returncode}",
                    "tune_before": dict(tune),
                }
            )
            if retries > args.max_retries:
                raise RuntimeError(
                    f"{stage}: exceeded max retries ({args.max_retries}) after runtime errors"
                ) from exc
            tune = retune(stage, tune, reason="runtime_error")
            continue

        stage_ckpt_dir = Path(args.run_root) / stage / "checkpoints"
        latest_any, latest_valid = latest_checkpoints(stage_ckpt_dir)
        if latest_valid is None:
            retries += 1
            gates.append(
                {
                    "gate": completed + gate_updates,
                    "decision": "restart_stage",
                    "reason": "no valid checkpoint",
                    "latest_any": str(latest_any) if latest_any else None,
                }
            )
            if retries > args.max_retries:
                raise RuntimeError(f"{stage}: exceeded max retries with invalid checkpoints")
            tune = retune(stage, tune, reason="gate_fail")
            continue

        vis_after = latest_visualization(stage, args.run_root)
        vis_after_mtime = vis_after.stat().st_mtime if vis_after else -1.0
        if vis_after is None or vis_after_mtime <= vis_before_mtime:
            retries += 1
            gates.append(
                {
                    "gate": completed + gate_updates,
                    "decision": "retune",
                    "reason": "no_new_visualization",
                    "visualization_before": str(vis_before) if vis_before else None,
                    "visualization_after": str(vis_after) if vis_after else None,
                    "tune_before": dict(tune),
                }
            )
            if retries > args.max_retries:
                raise RuntimeError(f"{stage}: exceeded max retries with stale visualizations")
            tune = retune(stage, tune, reason="gate_fail")
            continue

        run_dir = find_new_wandb_run(args.repo_root, start)
        if run_dir is None:
            raise RuntimeError(f"{stage}: no wandb run found after chunk")
        csv_path = export_stage_history(
            args.python_exec, args.repo_root, stage, run_dir, suffix=f"gate_{stage}"
        )
        if csv_path is None:
            raise RuntimeError(f"{stage}: failed to export gate CSV")
        verdict = gate_verdict(
            args.python_exec, args.repo_root, stage, csv_path, args.recent_rows
        )
        decision = verdict.get("decision", "retune")

        gate_info = {
            "gate": completed + gate_updates,
            "decision": decision,
            "csv": str(csv_path),
            "latest_valid_checkpoint": str(latest_valid),
            "latest_visualization": str(vis_after) if vis_after else None,
            "run_target_n_updates": run_target,
            "tune": dict(tune),
        }
        gates.append(gate_info)

        if decision in ("continue", "watch"):
            accepted_ckpt = str(latest_valid)
            completed += gate_updates
            retries = 0
            continue

        retries += 1
        if retries > args.max_retries:
            raise RuntimeError(f"{stage}: exceeded max retries ({args.max_retries})")
        tune = retune(stage, tune, reason="gate_fail")

    report["stages"][stage] = {
        "target_updates": target,
        "chunk_size": chunk,
        "initial_checkpoint": initial_ckpt_resolved,
        "final_checkpoint": accepted_ckpt,
        "gates": gates,
    }
    return accepted_ckpt


def write_report(report_path, report):
    lines = []
    lines.append(f"# TinyWorlds Auto Training Loop Report ({report['timestamp']})")
    lines.append("")
    lines.append(f"- run_root: `{report['run_root']}`")
    lines.append(f"- nproc_per_node: `{report['nproc_per_node']}`")
    lines.append("")
    for stage, data in report["stages"].items():
        lines.append(f"## {stage}")
        lines.append(f"- target_updates: `{data['target_updates']}`")
        lines.append(f"- chunk_size: `{data['chunk_size']}`")
        lines.append(f"- initial_checkpoint: `{data.get('initial_checkpoint')}`")
        lines.append(f"- final_checkpoint: `{data['final_checkpoint']}`")
        lines.append("- gates:")
        for g in data["gates"]:
            lines.append(
                f"  - gate `{g['gate']}` decision `{g['decision']}` checkpoint `{g.get('latest_valid_checkpoint')}`"
            )
        lines.append("")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Automate TinyWorlds gated training loop.")
    ap.add_argument("--repo-root", default=".", help="TinyWorlds repo root")
    ap.add_argument("--python-exec", default="./.venv/bin/python", help="Python interpreter path")
    ap.add_argument("--torchrun", default="./.venv/bin/torchrun", help="torchrun path")
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--video-config", default="configs/video_tokenizer.yaml")
    ap.add_argument("--latent-config", default="configs/latent_actions.yaml")
    ap.add_argument("--dynamics-config", default="configs/dynamics.yaml")
    ap.add_argument("--nproc-per-node", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--recent-rows", type=int, default=0, help="Gate scorer recent rows")
    ap.add_argument("--video-chunk", type=int, default=5000)
    ap.add_argument("--latent-chunk", type=int, default=1000)
    ap.add_argument("--dynamics-chunk", type=int, default=5000)
    ap.add_argument("--run-root", default="", help="results run root path override")
    ap.add_argument(
        "--target-updates",
        type=int,
        default=0,
        help="Override stage target n_updates (0 means use config value)",
    )
    ap.add_argument(
        "--init-learning-rate",
        type=float,
        default=None,
        help="Override initial learning_rate before auto retune",
    )
    ap.add_argument(
        "--init-grad-accum",
        type=int,
        default=None,
        help="Override initial gradient_accumulation_steps before auto retune",
    )
    ap.add_argument(
        "--init-batch-size",
        type=int,
        default=None,
        help="Override initial batch_size_per_gpu before auto retune",
    )
    ap.add_argument(
        "--init-log-interval",
        type=int,
        default=None,
        help="Override initial log_interval before auto retune",
    )
    ap.add_argument(
        "--only-stage",
        choices=["all", "video_tokenizer", "latent_actions", "dynamics"],
        default="all",
        help="Run the full pipeline or a single stage only",
    )
    ap.add_argument(
        "--video-checkpoint",
        default="",
        help="Video tokenizer checkpoint dir (or checkpoints root) for dynamics-only mode",
    )
    ap.add_argument(
        "--latent-checkpoint",
        default="",
        help="Latent actions checkpoint dir (or checkpoints root) for dynamics-only mode",
    )
    ap.add_argument(
        "--dynamics-checkpoint",
        default="",
        help="Optional dynamics checkpoint dir (or checkpoints root) to continue training",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    if not args.run_root:
        args.run_root = str(Path(args.repo_root) / "results" / f"auto_train_loop_{ts}")
    Path(args.run_root).mkdir(parents=True, exist_ok=True)
    args.chunk_size = {
        "video_tokenizer": args.video_chunk,
        "latent_actions": args.latent_chunk,
        "dynamics": args.dynamics_chunk,
    }

    report = {
        "timestamp": ts,
        "run_root": args.run_root,
        "nproc_per_node": args.nproc_per_node,
        "stages": {},
    }
    deps = {}
    if args.only_stage == "all":
        deps["video_tokenizer"] = run_stage(args, "video_tokenizer", deps, report)
        deps["latent_actions"] = run_stage(args, "latent_actions", deps, report)
        deps["dynamics"] = run_stage(args, "dynamics", deps, report)
    elif args.only_stage == "video_tokenizer":
        deps["video_tokenizer"] = run_stage(args, "video_tokenizer", deps, report)
    elif args.only_stage == "latent_actions":
        deps["latent_actions"] = run_stage(args, "latent_actions", deps, report)
    elif args.only_stage == "dynamics":
        if not args.video_checkpoint or not args.latent_checkpoint:
            raise ValueError(
                "--only-stage dynamics requires both --video-checkpoint and --latent-checkpoint"
            )
        deps["video_tokenizer"] = resolve_latest_valid_checkpoint(args.video_checkpoint)
        deps["latent_actions"] = resolve_latest_valid_checkpoint(args.latent_checkpoint)
        report["dependency_checkpoints"] = {
            "video_tokenizer": deps["video_tokenizer"],
            "latent_actions": deps["latent_actions"],
        }
        deps["dynamics"] = run_stage(
            args,
            "dynamics",
            deps,
            report,
            initial_checkpoint=args.dynamics_checkpoint if args.dynamics_checkpoint else None,
        )

    report_path = Path(args.repo_root) / "docs/action" / f"auto-training-loop-report-{ts}.md"
    write_report(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path), "deps": deps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
