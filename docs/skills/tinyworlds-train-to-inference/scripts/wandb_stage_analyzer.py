#!/usr/bin/env python3
"""Analyze stage metrics from W&B API (preferred) with local Logs fallback."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


STAGES = ["video_tokenizer", "latent_actions", "dynamics"]


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_stage_mapping(items):
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --stage-run-id value: {item}")
        stage, run_id = item.split("=", 1)
        stage = stage.strip()
        run_id = run_id.strip()
        if stage not in STAGES:
            raise ValueError(f"Unsupported stage in --stage-run-id: {stage}")
        if not run_id:
            raise ValueError(f"Empty run id in --stage-run-id: {item}")
        mapping[stage] = run_id
    return mapping


def read_csv_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def normalize_rows(rows):
    normalized = []
    for row in rows:
        normalized.append(dict(row))
    return normalized


def extract_series(rows, key):
    out = []
    for row in rows:
        out.append(to_float(row.get(key)))
    return out


def extract_steps(rows):
    steps = extract_series(rows, "_step")
    valid = []
    for idx, step in enumerate(steps):
        if step is None:
            continue
        valid.append((idx, int(step)))
    if valid:
        return valid
    return [(idx, idx) for idx, _ in enumerate(rows)]


def aligned_metric(rows, key, valid_index_step):
    values = []
    steps = []
    for idx, step in valid_index_step:
        value = to_float(rows[idx].get(key))
        if value is None:
            continue
        steps.append(step)
        values.append(value)
    return steps, values


def rolling_mean(values, window):
    if not values:
        return []
    window = max(1, int(window))
    out = []
    cur_sum = 0.0
    start = 0
    for idx, value in enumerate(values):
        cur_sum += value
        while idx - start + 1 > window:
            cur_sum -= values[start]
            start += 1
        out.append(cur_sum / (idx - start + 1))
    return out


def write_line_plot(path, title, x, ys, labels):
    fig, ax = plt.subplots(figsize=(10, 4))
    for y, label in zip(ys, labels):
        if y:
            ax.plot(x[: len(y)], y, label=label)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    if any(y for y in ys):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_dual_axis_plot(path, title, x1, y1, l1, x2, y2, l2):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_title(title)
    ax1.set_xlabel("step")
    ax1.grid(True, alpha=0.25)
    if y1:
        ax1.plot(x1[: len(y1)], y1, color="tab:blue", label=l1)
        ax1.set_ylabel(l1, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    if y2:
        ax2.plot(x2[: len(y2)], y2, color="tab:orange", label=l2)
        ax2.set_ylabel(l2, color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_bar_plot(path, title, labels, values):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["tab:green", "tab:red"])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def find_local_log_csv(repo_root, stage, run_id):
    logs_dir = Path(repo_root) / "Logs"
    if not logs_dir.exists():
        return None
    if run_id:
        matches = sorted(
            logs_dir.glob(f"{stage}_history_{run_id}_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if matches:
            return str(matches[-1].resolve())
    matches = sorted(
        logs_dir.glob(f"{stage}_history_*_gate_{stage}.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        return None
    return str(matches[-1].resolve())


def find_local_run_dir(repo_root, run_id):
    wandb_dir = Path(repo_root) / "wandb"
    if not wandb_dir.exists():
        return None
    matches = sorted(wandb_dir.glob(f"run-*-{run_id}"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def load_local_config_summary(repo_root, run_id):
    if not run_id:
        return {}, {}
    run_dir = find_local_run_dir(repo_root, run_id)
    if run_dir is None:
        return {}, {}

    config_file = run_dir / "files" / "config.yaml"
    summary_file = run_dir / "files" / "wandb-summary.json"
    config = {}
    summary = {}
    if config_file.is_file():
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        for key, value in raw.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value.get("value")
            else:
                config[key] = value
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return config, summary


def fetch_from_api(project, run_id):
    import wandb  # pylint: disable=import-outside-toplevel

    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    config = dict(run.config or {})
    summary = dict(getattr(run.summary, "_json_dict", {}) or {})
    rows = [dict(item) for item in run.scan_history()]
    entity = getattr(run, "entity", None)
    run_project = getattr(run, "project", project)
    url = f"https://wandb.ai/{entity}/{run_project}/runs/{run_id}" if entity else None
    return {
        "config": config,
        "summary": summary,
        "rows": rows,
        "url": url,
    }


def fetch_stage_data(args, stage, run_id):
    errors = []
    if args.wandb_source in ("api_first", "api_only") and run_id:
        try:
            api_data = fetch_from_api(args.wandb_project, run_id)
            if api_data["rows"]:
                return {"source": "api", "run_id": run_id, **api_data}
            errors.append(f"api rows empty for {stage}:{run_id}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"api error for {stage}:{run_id}: {exc}")
            if args.wandb_source == "api_only":
                return {"source": "api", "run_id": run_id, "error": "; ".join(errors)}

    if args.wandb_source in ("api_first", "local_only"):
        csv_path = find_local_log_csv(args.repo_root, stage, run_id)
        if csv_path:
            rows = normalize_rows(read_csv_rows(csv_path))
            config, summary = load_local_config_summary(args.repo_root, run_id)
            return {
                "source": "local",
                "run_id": run_id,
                "csv_path": csv_path,
                "rows": rows,
                "config": config,
                "summary": summary,
                "url": None,
            }
        errors.append(f"no local Logs csv for stage={stage} run_id={run_id}")
    return {"source": "none", "run_id": run_id, "error": "; ".join(errors)}


def summarize_config(config):
    keys = [
        "dataset",
        "batch_size_per_gpu",
        "gradient_accumulation_steps",
        "learning_rate",
        "n_updates",
        "embed_dim",
        "num_heads",
        "hidden_dim",
        "num_blocks",
        "patch_size",
        "context_length",
        "frame_size",
        "latent_dim",
        "num_bins",
        "n_actions",
        "use_actions",
    ]
    return {k: config.get(k) for k in keys if k in config}


def summarize_metrics(rows):
    valid = extract_steps(rows)
    _, losses = aligned_metric(rows, "train/loss", valid)
    _, lr = aligned_metric(rows, "learning_rate/group_0", valid)
    _, cbu1 = aligned_metric(rows, "train/codebook_usage", valid)
    _, cbu2 = aligned_metric(rows, "latent_actions/codebook_usage", valid)
    _, ent = aligned_metric(rows, "action_entropy", valid)
    _, uniq = aligned_metric(rows, "unique_actions", valid)
    out = {
        "rows": len(rows),
        "loss_final": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "lr_final": lr[-1] if lr else None,
        "codebook_usage_final": (cbu2[-1] if cbu2 else (cbu1[-1] if cbu1 else None)),
        "action_entropy_final": ent[-1] if ent else None,
        "unique_actions_final": uniq[-1] if uniq else None,
    }
    if losses:
        k = max(1, len(losses) // 10)
        out["loss_first10_mean"] = sum(losses[:k]) / k
        out["loss_last10_mean"] = sum(losses[-k:]) / k
    return out


def plot_stage(stage, rows, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    valid = extract_steps(rows)
    plot_paths = []

    sx, loss = aligned_metric(rows, "train/loss", valid)
    s_lr, lr = aligned_metric(rows, "learning_rate/group_0", valid)
    s_cbu_train, cbu_train = aligned_metric(rows, "train/codebook_usage", valid)
    s_cbu_latent, cbu_latent = aligned_metric(rows, "latent_actions/codebook_usage", valid)
    s_ent, ent = aligned_metric(rows, "action_entropy", valid)
    s_uniq, uniq = aligned_metric(rows, "unique_actions", valid)

    loss_plot = plot_dir / f"{stage}_loss.png"
    write_line_plot(loss_plot, f"{stage}: train/loss", sx, [loss], ["train/loss"])
    plot_paths.append(str(loss_plot))

    lr_plot = plot_dir / f"{stage}_learning_rate.png"
    write_line_plot(lr_plot, f"{stage}: learning_rate/group_0", s_lr, [lr], ["learning_rate/group_0"])
    plot_paths.append(str(lr_plot))

    if stage == "video_tokenizer":
        cbu_plot = plot_dir / f"{stage}_codebook_usage.png"
        write_line_plot(
            cbu_plot,
            f"{stage}: codebook usage",
            s_cbu_train,
            [cbu_train],
            ["train/codebook_usage"],
        )
        plot_paths.append(str(cbu_plot))
    elif stage == "latent_actions":
        cbu_plot = plot_dir / f"{stage}_codebook_usage.png"
        write_line_plot(
            cbu_plot,
            f"{stage}: latent_actions/codebook_usage",
            s_cbu_latent,
            [cbu_latent],
            ["latent_actions/codebook_usage"],
        )
        plot_paths.append(str(cbu_plot))

        au_plot = plot_dir / f"{stage}_entropy_unique_actions.png"
        write_dual_axis_plot(
            au_plot,
            f"{stage}: action_entropy vs unique_actions",
            s_ent,
            ent,
            "action_entropy",
            s_uniq,
            uniq,
            "unique_actions",
        )
        plot_paths.append(str(au_plot))
    elif stage == "dynamics":
        rm = rolling_mean(loss, max(3, len(loss) // 20)) if loss else []
        loss_roll_plot = plot_dir / f"{stage}_loss_rolling.png"
        write_line_plot(
            loss_roll_plot,
            f"{stage}: train/loss + rolling mean",
            sx,
            [loss, rm],
            ["train/loss", "rolling_mean"],
        )
        plot_paths.append(str(loss_roll_plot))

        if loss:
            k = max(1, len(loss) // 10)
            bar_plot = plot_dir / f"{stage}_loss_first10_vs_last10.png"
            first10 = sum(loss[:k]) / k
            last10 = sum(loss[-k:]) / k
            write_bar_plot(
                bar_plot,
                f"{stage}: loss first10% vs last10%",
                ["first10%", "last10%"],
                [first10, last10],
            )
            plot_paths.append(str(bar_plot))

        au_plot = plot_dir / f"{stage}_entropy_unique_actions.png"
        write_dual_axis_plot(
            au_plot,
            f"{stage}: action_entropy vs unique_actions",
            s_ent,
            ent,
            "action_entropy",
            s_uniq,
            uniq,
            "unique_actions",
        )
        plot_paths.append(str(au_plot))
    return plot_paths


def write_report(report_path, payload):
    lines = [
        f"# W&B Stage Analysis ({payload['timestamp']})",
        "",
        f"- run_root: `{payload['run_root']}`",
        f"- source_mode: `{payload['wandb_source']}`",
        f"- wandb_project: `{payload['wandb_project']}`",
        "",
    ]
    for stage in STAGES:
        data = payload["stages"].get(stage, {})
        lines += [f"## {stage}"]
        lines.append(f"- status: `{data.get('status')}`")
        lines.append(f"- source: `{data.get('source')}`")
        lines.append(f"- run_id: `{data.get('run_id')}`")
        lines.append(f"- run_url: `{data.get('url')}`")
        if data.get("csv_path"):
            lines.append(f"- csv_path: `{data['csv_path']}`")
        if data.get("error"):
            lines.append(f"- error: `{data['error']}`")

        metrics = data.get("metrics", {})
        if metrics:
            lines.append("- metrics:")
            for key in [
                "rows",
                "loss_final",
                "loss_min",
                "loss_first10_mean",
                "loss_last10_mean",
                "lr_final",
                "codebook_usage_final",
                "action_entropy_final",
                "unique_actions_final",
            ]:
                if key in metrics:
                    lines.append(f"  - `{key}`: `{metrics.get(key)}`")

        cfg = data.get("config_summary", {})
        if cfg:
            lines.append("- config:")
            for key, value in cfg.items():
                lines.append(f"  - `{key}`: `{value}`")

        plots = data.get("plots", [])
        if plots:
            lines.append("- plots:")
            for item in plots:
                lines.append(f"  - `{item}`")
        lines.append("")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stage metrics from W&B API/local Logs.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--wandb-project", default="tinyworlds")
    parser.add_argument(
        "--wandb-source",
        choices=["api_first", "api_only", "local_only"],
        default="api_first",
    )
    parser.add_argument("--analysis-out-dir", default="docs/action")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--stage-run-id", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = str(Path(args.repo_root).resolve())
    ts = args.timestamp or time.strftime("%Y_%m_%d_%H_%M_%S")
    mapping = parse_stage_mapping(args.stage_run_id)

    out_dir = Path(repo_root) / args.analysis_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_root = out_dir / "plots" / f"wandb-analysis-{ts}"
    plot_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": ts,
        "repo_root": repo_root,
        "run_root": str(Path(args.run_root).resolve()),
        "wandb_source": args.wandb_source,
        "wandb_project": args.wandb_project,
        "stages": {},
    }

    hard_fail = False
    for stage in STAGES:
        run_id = mapping.get(stage)
        data = fetch_stage_data(args, stage, run_id)
        if data.get("source") == "none" or not data.get("rows"):
            hard_fail = True
            payload["stages"][stage] = {
                "status": "missing",
                "source": data.get("source"),
                "run_id": run_id,
                "error": data.get("error"),
            }
            continue

        rows = data["rows"]
        plots = plot_stage(stage, rows, plot_root)
        metrics = summarize_metrics(rows)
        payload["stages"][stage] = {
            "status": "ok",
            "source": data.get("source"),
            "run_id": data.get("run_id"),
            "url": data.get("url"),
            "csv_path": data.get("csv_path"),
            "metrics": metrics,
            "config_summary": summarize_config(data.get("config", {})),
            "plots": plots,
        }

    report_path = out_dir / f"wandb-analysis-{ts}.md"
    json_path = out_dir / f"wandb-analysis-{ts}.json"
    write_report(report_path, payload)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "failed" if hard_fail else "ok",
                "report": str(report_path),
                "json": str(json_path),
                "plot_dir": str(plot_root),
                "stage_run_ids": mapping,
            },
            ensure_ascii=False,
        )
    )
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
