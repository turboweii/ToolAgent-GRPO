#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLE_GIT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "ToolAgent-GRPO-Eval")
ENTITY = os.environ.get("WANDB_ENTITY", "weixuan021103-tsinghua-university")
SETTINGS = wandb.Settings(save_code=False, disable_code=True, disable_git=True, x_save_requirements=False, x_disable_meta=True, x_disable_stats=True)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["covered_seen", "uncovered_seen", "unseen", "overall"]
METRICS = [
    ("pass_at_1", "pass_at_1"),
    ("pass_at_4", "pass_at_4"),
    ("pass_pow_4", "pass^4"),
    ("success_rate", "success_rate"),
    ("error_rate", "error_rate"),
    ("avg_tool_calls", "avg_tool_calls_per_sample"),
]

def split_block(report, split):
    return report if split == "overall" else report.get("by_split", {}).get(split, {})

def value_for(block, key):
    if key in block:
        return block[key]
    if key == "avg_tool_calls_per_sample" and "avg_tool_calls" in block:
        return block["avg_tool_calls"]
    return None

def main():
    run_name, method, rel_path = sys.argv[1], sys.argv[2], sys.argv[3]
    path = ROOT / rel_path
    report = json.loads(path.read_text())
    run = wandb.init(project=PROJECT, entity=ENTITY, name=run_name, config={"method": method, "source_json": rel_path}, settings=SETTINGS)
    payload = {}
    for split in SPLITS:
        block = split_block(report, split)
        for public_key, json_key in METRICS:
            v = value_for(block, json_key)
            if v is not None:
                payload[f"eval/{split}/{public_key}"] = float(v)
    run.log(payload, step=0)
    for k, v in payload.items():
        run.summary[k] = v
    print(run_name, run.url)
    run.finish()

if __name__ == "__main__":
    main()
