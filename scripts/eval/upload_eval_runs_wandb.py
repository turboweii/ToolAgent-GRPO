#!/usr/bin/env python3
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "ToolAgent-GRPO-Eval")
ENTITY = os.environ.get("WANDB_ENTITY", "weixuan021103-tsinghua-university")
SETTINGS = wandb.Settings(save_code=False)

ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("json_curriculum_previous", "curriculum", ROOT / "outputs/curriculum/20260703_0438_curriculum_previous_eval_report.json"),
    ("json_curriculum_repeat", "curriculum", ROOT / "outputs/curriculum/20260703_115619_curriculum_repeat_eval_report.json"),
    ("json_curriculum_unified", "curriculum", ROOT / "outputs/curriculum/20260703_131536_curriculum_unified_eval_report.json"),
    ("json_grpo_original", "grpo", ROOT / "outputs/grpo/20260701_grpo_original_eval_report.json"),
    ("json_grpo_previous_current", "grpo", ROOT / "outputs/grpo/20260703_115619_grpo_previous_current_eval_report.json"),
    ("json_grpo_repeat", "grpo", ROOT / "outputs/grpo/20260703_repeat_grpo_eval_report.json"),
]
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
    if split == "overall":
        return report
    return report.get("by_split", {}).get(split, {})


def value_for(block, key):
    if key in block:
        return block[key]
    if key == "avg_tool_calls_per_sample" and "avg_tool_calls" in block:
        return block["avg_tool_calls"]
    return None


def main():
    rows = []
    uploaded = []
    for run_name, method, path in RUNS:
        report = json.loads(path.read_text())
        config = {
            "method": method,
            "source_json": str(path.relative_to(ROOT)),
            "metric_definition": "pass_at_1 uses sample_id=0; pass_at_4 is any success among 4 samples; pass_pow_4 is all 4 samples successful.",
        }
        run = wandb.init(project=PROJECT, entity=ENTITY, name=run_name, config=config, reinit=True, settings=SETTINGS)
        payload = {}
        for split in SPLITS:
            block = split_block(report, split)
            row = {"run": run_name, "method": method, "split": split}
            for public_key, json_key in METRICS:
                v = value_for(block, json_key)
                if v is None:
                    continue
                v = float(v)
                metric_name = f"eval/{split}/{public_key}"
                payload[metric_name] = v
                row[public_key] = v
                run.summary[metric_name] = v
            rows.append(row)
        run.log(payload, step=0)
        uploaded.append(run.url)
        run.finish()

    table = wandb.Table(columns=["run", "method", "split"] + [m[0] for m in METRICS])
    for row in rows:
        table.add_data(*[row.get(c) for c in table.columns])
    run = wandb.init(project=PROJECT, entity=ENTITY, name="json_eval_summary_table", reinit=True, settings=SETTINGS)
    run.log({"eval/summary_table": table})
    uploaded.append(run.url)
    run.finish()
    print("uploaded runs:")
    for url in uploaded:
        print(url)


if __name__ == "__main__":
    main()

