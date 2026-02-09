#!/usr/bin/env python3
"""Audit 2025 TinyWorlds git history related to ZELDA train/inference."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


RELEVANT_PATTERN = re.compile(
    r"^(configs/(training|video_tokenizer|latent_actions|dynamics|inference|training_config|vqvae|lam)\.yaml"
    r"|configs/dev/dev_(training|training_config|pipeline)\.yaml"
    r"|scripts/(train_video_tokenizer|train_latent_actions|train_dynamics|run_inference|full_train)\.py"
    r"|run_inference(_teacher_forced)?\.py"
    r"|train_full_pipeline\.py"
    r"|src/(vqvae|latent_action_model|dynamics)/main\.py"
    r"|src/(vqvae|latent_action_model|dynamics)/train_.*\.py"
    r"|tests/test_inference.*\.py)$"
)

CURRENT_PATH_PATTERN = re.compile(
    r"^(configs/(training|video_tokenizer|latent_actions|dynamics|inference)\.yaml"
    r"|scripts/(train_video_tokenizer|train_latent_actions|train_dynamics|run_inference)\.py"
    r"|utils/inference_utils\.py"
    r"|utils/config\.py)$"
)


def run_capture(cmd, cwd):
    return subprocess.check_output(cmd, cwd=cwd, text=True)


def get_commits(repo_root, year):
    since = f"{year}-01-01"
    until = f"{year}-12-31 23:59:59"
    out = run_capture(["git", "rev-list", "--since", since, "--until", until, "HEAD"], cwd=repo_root)
    commits = [line.strip() for line in out.splitlines() if line.strip()]
    return commits


def changed_files(repo_root, commit):
    out = run_capture(["git", "show", "--pretty=", "--name-only", commit], cwd=repo_root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def has_zelda_in_changed_lines(repo_root, commit):
    out = run_capture(["git", "show", "--pretty=", "--unified=0", commit], cwd=repo_root)
    return bool(re.search(r"^[+-].*ZELDA", out, flags=re.MULTILINE))


def has_zelda_anywhere(repo_root, commit):
    out = run_capture(["git", "show", "--pretty=", commit], cwd=repo_root)
    return "ZELDA" in out


def commit_subject(repo_root, commit):
    return run_capture(
        ["git", "show", "-s", "--format=%h %ad %s", "--date=short", commit], cwd=repo_root
    ).strip()


def select_mode(repo_root, commits, mode):
    if mode not in ("strict", "wide", "current_paths"):
        raise ValueError(f"Unsupported mode: {mode}")

    selected = []
    unique_files = set()
    touches = 0
    pattern = RELEVANT_PATTERN if mode in ("strict", "wide") else CURRENT_PATH_PATTERN

    for commit in commits:
        files = changed_files(repo_root, commit)
        relevant = [f for f in files if pattern.match(f)]
        if not relevant:
            continue

        if mode == "strict":
            zelda_hit = has_zelda_in_changed_lines(repo_root, commit)
        else:
            zelda_hit = has_zelda_anywhere(repo_root, commit)
        if not zelda_hit:
            continue

        selected.append(commit)
        touches += len(relevant)
        unique_files.update(relevant)

    selected_subjects = [commit_subject(repo_root, c) for c in selected]
    selected_subjects.sort()
    return {
        "mode": mode,
        "selected_commits": selected,
        "selected_subjects": selected_subjects,
        "selected_count": len(selected),
        "relevant_file_touches": touches,
        "unique_files": sorted(unique_files),
    }


def write_markdown(out_path, payload):
    lines = [
        f"# ZELDA Git Audit ({payload['timestamp']})",
        "",
        f"- year: `{payload['year']}`",
        f"- mode: `{payload['mode']}`",
        f"- total_commits_in_year: `{payload['total_commits_in_year']}`",
        f"- selected_commits: `{payload['selected_count']}`",
        f"- relevant_file_touches: `{payload['relevant_file_touches']}`",
        "",
        "## Selected Commits",
    ]
    for item in payload["selected_subjects"]:
        lines.append(f"- {item}")
    lines += ["", "## Unique Relevant Files"]
    for file_path in payload["unique_files"]:
        lines.append(f"- `{file_path}`")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit git history for ZELDA train/inference traces.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--mode", choices=["strict", "wide", "current_paths"], default="strict")
    parser.add_argument("--out-dir", default="docs/action")
    parser.add_argument("--timestamp", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = str(Path(args.repo_root).resolve())
    ts = args.timestamp or time.strftime("%Y_%m_%d_%H_%M_%S")
    commits = get_commits(repo_root, args.year)
    base = select_mode(repo_root, commits, args.mode)

    payload = {
        "timestamp": ts,
        "year": args.year,
        "mode": args.mode,
        "total_commits_in_year": len(commits),
        "selected_count": base["selected_count"],
        "relevant_file_touches": base["relevant_file_touches"],
        "selected_subjects": base["selected_subjects"],
        "selected_commits": base["selected_commits"],
        "unique_files": base["unique_files"],
    }
    out_dir = Path(repo_root) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"zelda-{args.year}-git-evidence-{ts}.md"
    write_markdown(report_path, payload)

    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "year": args.year,
                "mode": args.mode,
                "total_commits_in_year": payload["total_commits_in_year"],
                "selected_count": payload["selected_count"],
                "relevant_file_touches": payload["relevant_file_touches"],
                "selected_commits": payload["selected_commits"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
