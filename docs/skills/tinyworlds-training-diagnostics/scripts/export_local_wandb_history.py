#!/usr/bin/env python3
"""Export local wandb .wandb history records into JSON/CSV files.

Usage:
  python export_local_wandb_history.py \
    --run "video_tokenizer=wandb/run-20260204_195233-okmjny24/run-okmjny24.wandb" \
    --run "latent_actions=wandb/run-20260205_210649-d376skck/run-d376skck.wandb" \
    --run "dynamics=wandb/run-20260206_004434-2pghxx71/run-2pghxx71.wandb" \
    --out-dir Logs \
    --suffix latest
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


def _parse_json_value(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _assign_nested(obj: dict, keys, value):
    if len(keys) == 1:
        obj[keys[0]] = value
        return

    cur = obj
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _extract_history(run_file: Path):
    ds = datastore.DataStore()
    ds.open_for_scan(str(run_file))

    rows = []
    while True:
        blob = ds.scan_data()
        if blob is None:
            break

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(blob)
        if rec.WhichOneof("record_type") != "history":
            continue

        row = {}
        for item in rec.history.item:
            keys = list(item.nested_key)
            if not keys:
                continue
            value = _parse_json_value(item.value_json)
            _assign_nested(row, keys, value)
        if row:
            rows.append(row)

    if rows and all(isinstance(r.get("_step"), (int, float)) for r in rows if "_step" in r):
        rows.sort(key=lambda x: x.get("_step", -1))
    return rows


def _to_csv(rows, out_csv: Path):
    fields = set()
    for row in rows:
        fields.update(row.keys())

    preferred = ["_runtime", "_step", "_timestamp", "train/loss", "train/learning_rate"]
    columns = [c for c in preferred if c in fields] + sorted(c for c in fields if c not in preferred)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {}
            for c in columns:
                val = row.get(c, "")
                if isinstance(val, (dict, list)):
                    out[c] = json.dumps(val, ensure_ascii=False)
                else:
                    out[c] = val
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Export local wandb history to Logs JSON/CSV.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Mapping in form stage=path/to/run-<id>.wandb (repeatable)",
    )
    parser.add_argument("--out-dir", default="Logs", help="Output directory")
    parser.add_argument("--suffix", default="latest", help="Filename suffix")
    args = parser.parse_args()

    mappings = OrderedDict()
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run value: {item}. Expected stage=path")
        stage, path = item.split("=", 1)
        stage = stage.strip()
        run_file = Path(path.strip())
        if not stage:
            raise ValueError(f"Invalid stage in --run value: {item}")
        if not run_file.exists():
            raise FileNotFoundError(f"Run file not found: {run_file}")
        mappings[stage] = run_file

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stage, run_file in mappings.items():
        run_id = run_file.stem.replace("run-", "")
        rows = _extract_history(run_file)

        out_json = out_dir / f"{stage}_history_{run_id}_{args.suffix}.json"
        out_csv = out_dir / f"{stage}_history_{run_id}_{args.suffix}.csv"

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        _to_csv(rows, out_csv)

        print(f"{stage}\t{run_id}\trows={len(rows)}\tjson={out_json}\tcsv={out_csv}")


if __name__ == "__main__":
    main()
