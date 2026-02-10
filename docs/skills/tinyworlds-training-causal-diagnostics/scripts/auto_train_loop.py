#!/usr/bin/env python3
"""Automate TinyWorlds gated training with retune/retry loops."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

STAGE_SCRIPT_NAMES = {
    "video_tokenizer": "train_video_tokenizer.py",
    "latent_actions": "train_latent_actions.py",
    "dynamics": "train_dynamics.py",
}


def parse_bool(text):
    low = str(text).strip().lower()
    if low in ("1", "true", "yes", "y", "on"):
        return True
    if low in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {text}")


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


def safe_check_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def list_process_table():
    out = safe_check_output(["ps", "-eo", "pid=,ppid=,stat=,cmd="])
    table = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = {
            "pid": pid,
            "ppid": ppid,
            "state": parts[2],
            "cmd": parts[3],
        }
    return table


def descendants_of(root_pid, table):
    by_parent = {}
    for pid, row in table.items():
        by_parent.setdefault(row["ppid"], []).append(pid)
    out = []
    stack = list(by_parent.get(root_pid, []))
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(by_parent.get(pid, []))
    return out


def pid_exists(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_process_tree(root_pid, grace_seconds=10.0):
    table = list_process_table()
    tree = [root_pid] + descendants_of(root_pid, table)
    tree = [pid for pid in sorted(set(tree), reverse=True) if pid != os.getpid()]
    if not tree:
        return {"root_pid": root_pid, "targeted": [], "remaining": []}

    for pid in tree:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    deadline = time.time() + max(0.0, float(grace_seconds))
    while time.time() < deadline:
        remaining = [pid for pid in tree if pid_exists(pid)]
        if not remaining:
            return {"root_pid": root_pid, "targeted": tree, "remaining": []}
        time.sleep(0.5)

    remaining = [pid for pid in tree if pid_exists(pid)]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(0.5)
    still = [pid for pid in remaining if pid_exists(pid)]
    return {"root_pid": root_pid, "targeted": tree, "remaining": still}


def stage_from_cmd(cmd):
    cmd_low = cmd.lower()
    for stage, script_name in STAGE_SCRIPT_NAMES.items():
        if script_name in cmd_low:
            return stage
    return None


def is_stage_launcher(cmd):
    stage = stage_from_cmd(cmd)
    if stage is None:
        return False, None
    cmd_low = cmd.lower()
    is_torchrun = "torchrun" in cmd_low and STAGE_SCRIPT_NAMES[stage] in cmd_low
    is_direct_python = (
        "python" in cmd_low
        and STAGE_SCRIPT_NAMES[stage] in cmd_low
        and "--local-rank" not in cmd_low
        and "--local_rank" not in cmd_low
    )
    return (is_torchrun or is_direct_python), stage


def find_training_launchers(stage=None):
    rows = []
    for row in list_process_table().values():
        keep, row_stage = is_stage_launcher(row["cmd"])
        if not keep:
            continue
        if stage is not None and row_stage != stage:
            continue
        row = dict(row)
        row["stage"] = row_stage
        rows.append(row)
    rows.sort(key=lambda x: x["pid"])
    return rows


def find_training_stage_processes(table=None):
    if table is None:
        table = list_process_table()
    rows = []
    for row in table.values():
        row_stage = stage_from_cmd(row["cmd"])
        if row_stage is None:
            continue
        item = dict(row)
        item["stage"] = row_stage
        rows.append(item)
    rows.sort(key=lambda x: x["pid"])
    return rows


def cleanup_extra_launchers(keep_pid=None):
    table = list_process_table()
    launchers = find_training_launchers(stage=None)
    stage_rows = find_training_stage_processes(table=table)
    keep_tree = set()
    if keep_pid is not None:
        keep_tree = set([int(keep_pid)] + descendants_of(int(keep_pid), table))

    extras = [
        row
        for row in stage_rows
        if row["pid"] not in keep_tree and row["pid"] != os.getpid()
    ]
    extra_pids = {row["pid"] for row in extras}
    roots = [row for row in extras if row["ppid"] not in extra_pids]

    killed = []
    for row in roots:
        pid = int(row["pid"])
        result = terminate_process_tree(pid)
        killed.append(
            {
                "pid": pid,
                "stage": row["stage"],
                "state": row["state"],
                "remaining": result["remaining"],
            }
        )
    return {
        "seen_launchers": launchers,
        "seen_stage_processes": stage_rows,
        "killed": killed,
    }


def query_gpu_utilization():
    out = safe_check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    values = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            values[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return values


def query_pid_to_gpu_indices():
    gpu_uuid_out = safe_check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
    )
    uuid_to_idx = {}
    for raw in gpu_uuid_out.splitlines():
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            uuid_to_idx[parts[1]] = int(parts[0])
        except ValueError:
            continue

    app_out = safe_check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    pid_map = {}
    for raw in app_out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_uuid = parts[0]
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        gpu_idx = uuid_to_idx.get(gpu_uuid)
        if gpu_idx is None:
            continue
        pid_map.setdefault(pid, set()).add(gpu_idx)
    return pid_map


def training_gpu_indices_for_launcher(launcher_pid):
    table = list_process_table()
    training_tree = set([launcher_pid] + descendants_of(launcher_pid, table))
    pid_to_gpu = query_pid_to_gpu_indices()
    gpus = set()
    for pid in training_tree:
        gpus.update(pid_to_gpu.get(pid, set()))
    return sorted(gpus)


def run_training_with_monitor(cmd, env, stage, args):
    monitor = {
        "stage": stage,
        "launcher_pid": None,
        "cleanup_events": [],
        "gpu_samples": 0,
        "gpu_util_max": {},
        "gpu_util_hits": {},
        "gpu_process_gpu_max": 0,
        "gpu_process_dual_samples": 0,
        "dual_gpu_check_enabled": bool(
            args.enforce_dual_gpu_check and int(args.nproc_per_node) >= 2
        ),
        "dual_gpu_ok": True,
    }

    if args.cleanup_extra_processes:
        pre = cleanup_extra_launchers(keep_pid=None)
        if pre["killed"]:
            monitor["cleanup_events"].append({"phase": "pre_launch", **pre})

    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )
    monitor["launcher_pid"] = proc.pid

    interval = max(0.5, float(args.monitor_interval_sec))
    while True:
        rc = proc.poll()

        if args.cleanup_extra_processes:
            running = cleanup_extra_launchers(keep_pid=proc.pid)
            if running["killed"]:
                monitor["cleanup_events"].append({"phase": "running", **running})

        if monitor["dual_gpu_check_enabled"]:
            util = query_gpu_utilization()
            gpu_indices = training_gpu_indices_for_launcher(proc.pid)
            monitor["gpu_samples"] += 1
            monitor["gpu_process_gpu_max"] = max(
                monitor["gpu_process_gpu_max"], len(gpu_indices)
            )
            if len(gpu_indices) >= 2:
                monitor["gpu_process_dual_samples"] += 1
            for idx, u in util.items():
                key = str(idx)
                monitor["gpu_util_max"][key] = max(
                    float(u), float(monitor["gpu_util_max"].get(key, 0.0))
                )
                if float(u) >= float(args.gpu_util_threshold):
                    monitor["gpu_util_hits"][key] = int(monitor["gpu_util_hits"].get(key, 0)) + 1

        if rc is not None:
            break
        time.sleep(interval)

    if monitor["dual_gpu_check_enabled"]:
        util_ok_gpus = [
            idx
            for idx, hits in monitor["gpu_util_hits"].items()
            if int(hits) >= int(args.gpu_required_samples)
        ]
        process_ok = monitor["gpu_process_gpu_max"] >= 2 or monitor["gpu_process_dual_samples"] >= 1
        util_ok = len(util_ok_gpus) >= 2
        monitor["dual_gpu_ok"] = bool(process_ok or util_ok)
        monitor["dual_gpu_detail"] = {
            "required_samples": int(args.gpu_required_samples),
            "util_threshold": float(args.gpu_util_threshold),
            "util_ok_gpus": util_ok_gpus,
            "process_ok": process_ok,
            "util_ok": util_ok,
        }

    if args.cleanup_extra_processes:
        post = cleanup_extra_launchers(keep_pid=None)
        if post["killed"]:
            monitor["cleanup_events"].append({"phase": "post_run", **post})
    return int(proc.returncode), monitor


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


def wandb_run_id_from_dir(run_dir):
    if run_dir is None:
        return None
    parts = str(run_dir).rstrip("/").split("-")
    return parts[-1] if parts else None


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
    return f"scripts/{STAGE_SCRIPT_NAMES[stage]}"


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
    if reason == "gpu_guard_fail":
        # Process/GPU guard failures are usually orchestration issues; keep optimizer knobs unchanged.
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
    retries = 0
    accepted_ckpt = None
    if initial_checkpoint:
        resolved = resolve_latest_valid_checkpoint(initial_checkpoint)
        accepted_ckpt = resolved
    initial_ckpt_resolved = accepted_ckpt
    completed = 0
    if accepted_ckpt:
        ckpt_step = step_from_name(Path(accepted_ckpt).name)
        if ckpt_step >= 0:
            completed = ckpt_step + 1
    initial_completed_updates = completed
    gates = []
    accepted_run_id = None

    while completed < target:
        gate_updates = min(chunk, target - completed)
        run_target = completed + gate_updates
        vis_before = latest_visualization(stage, args.run_root)
        vis_before_mtime = vis_before.stat().st_mtime if vis_before else -1.0
        if int(args.nproc_per_node) <= 1:
            cmd = [
                args.python_exec,
                stage_script(stage),
                "--config",
                stage_config(stage, args),
                "--training_config",
                args.training_config,
                "distributed.use_ddp=false",
                "nproc_per_node=1",
                f"n_updates={run_target}",
                f"batch_size_per_gpu={tune['batch_size_per_gpu']}",
                f"learning_rate={tune['learning_rate']}",
                f"gradient_accumulation_steps={tune['gradient_accumulation_steps']}",
                f"log_interval={tune['log_interval']}",
            ]
        else:
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
        returncode, runtime_monitor = run_training_with_monitor(cmd, env=env, stage=stage, args=args)
        if returncode != 0:
            retries += 1
            gates.append(
                {
                    "gate": completed + gate_updates,
                    "decision": "retune",
                    "reason": f"runtime_error_exit_{returncode}",
                    "tune_before": dict(tune),
                    "runtime_monitor": runtime_monitor,
                }
            )
            if retries > args.max_retries:
                raise RuntimeError(
                    f"{stage}: exceeded max retries ({args.max_retries}) after runtime errors"
                )
            tune = retune(stage, tune, reason="runtime_error")
            continue
        if runtime_monitor.get("dual_gpu_check_enabled") and not runtime_monitor.get("dual_gpu_ok"):
            retries += 1
            gates.append(
                {
                    "gate": completed + gate_updates,
                    "decision": "retune",
                    "reason": "dual_gpu_not_active",
                    "tune_before": dict(tune),
                    "runtime_monitor": runtime_monitor,
                }
            )
            if retries > args.max_retries:
                raise RuntimeError(
                    f"{stage}: exceeded max retries ({args.max_retries}) after dual-GPU guard failures"
                )
            tune = retune(stage, tune, reason="gpu_guard_fail")
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
            "gate": run_target,
            "decision": decision,
            "csv": str(csv_path),
            "latest_valid_checkpoint": str(latest_valid),
            "latest_visualization": str(vis_after) if vis_after else None,
            "run_target_n_updates": run_target,
            "tune": dict(tune),
            "wandb_run_dir": str(run_dir),
            "wandb_run_id": wandb_run_id_from_dir(run_dir),
            "runtime_monitor": runtime_monitor,
        }
        gates.append(gate_info)

        if decision in ("continue", "watch"):
            accepted_ckpt = str(latest_valid)
            accepted_run_id = gate_info["wandb_run_id"]
            latest_step = step_from_name(Path(accepted_ckpt).name)
            if latest_step >= 0:
                completed = max(completed + gate_updates, latest_step + 1)
            else:
                completed += gate_updates
            retries = 0
            continue

        retries += 1
        if retries > args.max_retries:
            raise RuntimeError(f"{stage}: exceeded max retries ({args.max_retries})")
        tune = retune(stage, tune, reason="gate_fail")

    report["stages"][stage] = {
        "target_updates": target,
        "initial_completed_updates": initial_completed_updates,
        "chunk_size": chunk,
        "initial_checkpoint": initial_ckpt_resolved,
        "final_checkpoint": accepted_ckpt,
        "stage_run_id": accepted_run_id,
        "gates": gates,
    }
    return accepted_ckpt


def write_report(report_path, report):
    lines = []
    lines.append(f"# TinyWorlds Auto Training Loop Report ({report['timestamp']})")
    lines.append("")
    lines.append(f"- run_root: `{report['run_root']}`")
    lines.append(f"- nproc_per_node: `{report['nproc_per_node']}`")
    lines.append(f"- cleanup_extra_processes: `{report.get('cleanup_extra_processes')}`")
    lines.append(f"- enforce_dual_gpu_check: `{report.get('enforce_dual_gpu_check')}`")
    lines.append(f"- monitor_interval_sec: `{report.get('monitor_interval_sec')}`")
    lines.append("")
    for stage, data in report["stages"].items():
        lines.append(f"## {stage}")
        lines.append(f"- target_updates: `{data['target_updates']}`")
        lines.append(f"- initial_completed_updates: `{data.get('initial_completed_updates', 0)}`")
        lines.append(f"- chunk_size: `{data['chunk_size']}`")
        lines.append(f"- initial_checkpoint: `{data.get('initial_checkpoint')}`")
        lines.append(f"- final_checkpoint: `{data['final_checkpoint']}`")
        lines.append(f"- stage_run_id: `{data.get('stage_run_id')}`")
        lines.append("- gates:")
        for g in data["gates"]:
            monitor = g.get("runtime_monitor") or {}
            dual = monitor.get("dual_gpu_ok")
            launcher = monitor.get("launcher_pid")
            lines.append(
                f"  - gate `{g['gate']}` decision `{g['decision']}` checkpoint `{g.get('latest_valid_checkpoint')}` launcher `{launcher}` dual_gpu_ok `{dual}`"
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
    ap.add_argument(
        "--cleanup-extra-processes",
        type=parse_bool,
        default=True,
        help="Detect and terminate extra stage launchers; keep only current training process",
    )
    ap.add_argument(
        "--enforce-dual-gpu-check",
        type=parse_bool,
        default=True,
        help="When nproc_per_node>=2, require evidence that both GPUs were active during training",
    )
    ap.add_argument(
        "--monitor-interval-sec",
        type=float,
        default=5.0,
        help="Seconds between process/GPU monitor checks while a stage is running",
    )
    ap.add_argument(
        "--gpu-util-threshold",
        type=float,
        default=1.0,
        help="GPU utilization threshold (percent) counted as active sample",
    )
    ap.add_argument(
        "--gpu-required-samples",
        type=int,
        default=2,
        help="Required active samples per GPU for utilization-based dual-GPU evidence",
    )
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
        "cleanup_extra_processes": bool(args.cleanup_extra_processes),
        "enforce_dual_gpu_check": bool(args.enforce_dual_gpu_check),
        "monitor_interval_sec": float(args.monitor_interval_sec),
        "stages": {},
    }
    deps = {}
    if args.only_stage == "all":
        deps["video_tokenizer"] = run_stage(
            args,
            "video_tokenizer",
            deps,
            report,
            initial_checkpoint=args.video_checkpoint if args.video_checkpoint else None,
        )
        deps["latent_actions"] = run_stage(
            args,
            "latent_actions",
            deps,
            report,
            initial_checkpoint=args.latent_checkpoint if args.latent_checkpoint else None,
        )
        deps["dynamics"] = run_stage(
            args,
            "dynamics",
            deps,
            report,
            initial_checkpoint=args.dynamics_checkpoint if args.dynamics_checkpoint else None,
        )
    elif args.only_stage == "video_tokenizer":
        deps["video_tokenizer"] = run_stage(
            args,
            "video_tokenizer",
            deps,
            report,
            initial_checkpoint=args.video_checkpoint if args.video_checkpoint else None,
        )
    elif args.only_stage == "latent_actions":
        deps["latent_actions"] = run_stage(
            args,
            "latent_actions",
            deps,
            report,
            initial_checkpoint=args.latent_checkpoint if args.latent_checkpoint else None,
        )
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
    stage_run_ids = {
        stage: data.get("stage_run_id")
        for stage, data in report["stages"].items()
        if data.get("stage_run_id")
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "deps": deps,
                "stage_run_ids": stage_run_ids,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
