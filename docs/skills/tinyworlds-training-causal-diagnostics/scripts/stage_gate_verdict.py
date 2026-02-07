#!/usr/bin/env python3
"""Compute gate metrics and verdict for TinyWorlds stage CSV history."""

import argparse
import csv
import json
import math
from statistics import median


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = to_float(row.get("_step"))
            loss = to_float(row.get("train/loss"))
            if step is None or loss is None:
                continue
            rows.append(
                {
                    "step": int(step),
                    "loss": loss,
                    "action_entropy": to_float(row.get("action_entropy")),
                    "unique_actions": to_float(row.get("unique_actions")),
                    "codebook_usage": to_float(
                        row.get("latent_actions/codebook_usage")
                        or row.get("train/codebook_usage")
                    ),
                    "learning_rate": to_float(row.get("learning_rate/group_0")),
                }
            )
    rows.sort(key=lambda x: x["step"])
    return rows


def thirds(vals):
    n = len(vals)
    if n < 3:
        return None
    return vals[: n // 3], vals[n // 3 : 2 * n // 3], vals[2 * n // 3 :]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows):
    losses = [r["loss"] for r in rows]
    out = {
        "rows": len(rows),
        "step_min": rows[0]["step"],
        "step_max": rows[-1]["step"],
        "loss_final": losses[-1],
        "loss_min": min(losses),
        "loss_median": median(losses),
        "loss_first": losses[0],
        "loss_drop_pct_from_first": (losses[0] - losses[-1]) * 100.0 / losses[0]
        if losses[0] != 0
        else None,
    }
    segs = thirds(rows)
    if segs:
        e, m, l = segs
        out["loss_mean_early"] = mean([x["loss"] for x in e])
        out["loss_mean_mid"] = mean([x["loss"] for x in m])
        out["loss_mean_late"] = mean([x["loss"] for x in l])
        k = max(1, len(rows) // 10)
        first10 = mean([x["loss"] for x in rows[:k]])
        last10 = mean([x["loss"] for x in rows[-k:]])
        out["loss_first10_mean"] = first10
        out["loss_last10_mean"] = last10
        out["loss_delta_last10_minus_first10"] = (
            None if first10 is None or last10 is None else (last10 - first10)
        )

    ent = [r["action_entropy"] for r in rows if r["action_entropy"] is not None]
    ua = [r["unique_actions"] for r in rows if r["unique_actions"] is not None]
    cbu = [r["codebook_usage"] for r in rows if r["codebook_usage"] is not None]
    lr = [r["learning_rate"] for r in rows if r["learning_rate"] is not None]
    if ent:
        out["action_entropy_final"] = ent[-1]
        out["action_entropy_median"] = median(ent)
    if ua:
        out["unique_actions_final"] = ua[-1]
        out["unique_actions_median"] = median(ua)
    if cbu:
        out["codebook_usage_final"] = cbu[-1]
        out["codebook_usage_median"] = median(cbu)
    if lr:
        out["learning_rate_final"] = lr[-1]
        out["learning_rate_max"] = max(lr)
    return out


def verdict_video(s):
    pass_ok = (
        s["loss_final"] <= 0.01
        and s.get("loss_mean_late", s["loss_final"])
        <= s.get("loss_mean_mid", s["loss_final"]) * 1.05
    )
    watch = s["loss_final"] <= 0.02
    if pass_ok:
        return "continue"
    if watch:
        return "watch"
    return "retune"


def verdict_latent(s):
    lf = s["loss_final"]
    ua = s.get("unique_actions_final", 0.0) or 0.0
    ent = s.get("action_entropy_final", 0.0) or 0.0
    if lf <= 0.06 and ua >= 2 and ent >= 0.4:
        return "continue"
    if lf <= 0.1 and ua >= 2:
        return "watch"
    return "retune"


def verdict_dynamics(s):
    mid = s.get("loss_mean_mid")
    late = s.get("loss_mean_late")
    delta = s.get("loss_delta_last10_minus_first10")
    final = s.get("loss_final")
    if final is not None and final >= 6.0 and (delta or 0.0) >= 0.0:
        return "retune"
    if (
        mid is not None
        and late is not None
        and final is not None
        and late <= mid * 1.02
        and (delta or 0.0) <= 0.2
        and final <= 5.0
    ):
        return "continue"
    if mid is not None and late is not None and late <= mid * 1.08 and (delta or 0.0) <= 0.5:
        return "watch"
    return "retune"


def choose_verdict(stage, summary):
    if stage == "video_tokenizer":
        return verdict_video(summary)
    if stage == "latent_actions":
        return verdict_latent(summary)
    if stage == "dynamics":
        return verdict_dynamics(summary)
    raise ValueError(f"Unsupported stage: {stage}")


def main():
    ap = argparse.ArgumentParser(description="Gate verdict from TinyWorlds stage CSV.")
    ap.add_argument("--stage", choices=["video_tokenizer", "latent_actions", "dynamics"], required=True)
    ap.add_argument("--csv", required=True, help="History CSV file path")
    ap.add_argument(
        "--recent-rows",
        type=int,
        default=0,
        help="Use only last N rows (0 means all rows)",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("No valid rows in CSV")
    if args.recent_rows > 0 and len(rows) > args.recent_rows:
        rows = rows[-args.recent_rows :]

    s = summarize(rows)
    decision = choose_verdict(args.stage, s)
    out = {
        "stage": args.stage,
        "decision": decision,
        "summary": s,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
