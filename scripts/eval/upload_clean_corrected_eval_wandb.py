#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

RUNS = [
    ("curriculum", "curriculum previous", "outputs/curriculum/20260703_0438_curriculum_previous_eval_report.json"),
    ("curriculum", "curriculum repeat", "outputs/curriculum/20260703_115619_curriculum_repeat_eval_report.json"),
    ("curriculum", "curriculum unified", "outputs/curriculum/20260703_131536_curriculum_unified_eval_report.json"),
    ("ordinary_grpo", "ordinary original", "outputs/grpo/20260701_grpo_original_eval_report.json"),
    ("ordinary_grpo", "ordinary previous/current", "outputs/grpo/20260703_115619_grpo_previous_current_eval_report.json"),
    ("ordinary_grpo", "ordinary repeat", "outputs/grpo/20260703_repeat_grpo_eval_report.json"),
]
SPLITS = ["covered_seen", "uncovered_seen", "unseen", "overall"]
METRICS = [
    ("pass_at_1", "pass@1", "score"),
    ("pass_at_4", "pass@4", "score"),
    ("pass^4", "pass^4", "score"),
    ("success_rate", "success rate", "rate"),
    ("error_rate", "error rate", "rate"),
    ("avg_tool_calls_per_sample", "avg tool calls", "calls/sample"),
]

COLORS = {
    "curriculum previous": "#4C78A8",
    "curriculum repeat": "#72B7B2",
    "curriculum unified": "#54A24B",
    "ordinary original": "#F58518",
    "ordinary previous/current": "#E45756",
    "ordinary repeat": "#B279A2",
    "Ordinary GRPO avg": "#F58518",
    "Curriculum avg": "#54A24B",
}

def split_metric(split_data, key):
    if key == "avg_tool_calls_per_sample":
        return float(split_data.get("avg_tool_calls_per_sample", split_data.get("avg_tool_calls", 0.0)))
    return float(split_data[key])

def load_rows(root: Path):
    rows = []
    for group, run, rel in RUNS:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
        by_split = data["by_split"]
        for split in SPLITS:
            sd = by_split[split]
            row = {"group": group, "run": run, "split": split, "path": rel}
            for key, _, _ in METRICS:
                row[key] = split_metric(sd, key)
            rows.append(row)
    return rows

def average_rows(rows):
    out = []
    for group, name in [("ordinary_grpo", "Ordinary GRPO avg"), ("curriculum", "Curriculum avg")]:
        for split in SPLITS:
            items = [r for r in rows if r["group"] == group and r["split"] == split]
            row = {"group": group, "run": name, "split": split, "path": "average over 3 runs"}
            for key, _, _ in METRICS:
                row[key] = mean(r[key] for r in items)
            out.append(row)
    return out

def save_clean_plots(rows, avg_rows, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for key, title, ylabel in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [2.2, 1.0]})

        # Left: six individual runs, grouped by split.
        ax = axes[0]
        run_names = [r[1] for r in RUNS]
        x = list(range(len(SPLITS)))
        width = 0.12
        offsets = [(-2.5 + i) * width for i in range(len(run_names))]
        for offset, run in zip(offsets, run_names):
            vals = [next(r for r in rows if r["run"] == run and r["split"] == split)[key] for split in SPLITS]
            ax.bar([i + offset for i in x], vals, width, label=run, color=COLORS[run])
        ax.set_title(f"{title}: six eval runs")
        ax.set_xticks(x, SPLITS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        if key != "avg_tool_calls_per_sample":
            ax.set_ylim(0, 0.9 if key in {"success_rate", "error_rate"} else 0.55)

        # Right: 3-run averages.
        ax = axes[1]
        avg_names = ["Ordinary GRPO avg", "Curriculum avg"]
        width2 = 0.35
        for j, run in enumerate(avg_names):
            vals = [next(r for r in avg_rows if r["run"] == run and r["split"] == split)[key] for split in SPLITS]
            ax.bar([i + (j - 0.5) * width2 for i in x], vals, width2, label=run, color=COLORS[run])
        ax.set_title(f"{title}: 3-run average")
        ax.set_xticks(x, SPLITS, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
        if key != "avg_tool_calls_per_sample":
            ax.set_ylim(0, 0.9 if key in {"success_rate", "error_rate"} else 0.55)

        fig.suptitle(f"Corrected ToolAgent-GRPO Evaluation - {title}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = out_dir / f"clean_{key.replace('^','pow').replace('@','at')}_split_comparison.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append((key, title, path))
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--entity", default="weixuan021103-tsinghua-university")
    parser.add_argument("--project", default="ToolAgent-GRPO-Eval")
    parser.add_argument("--run-name", default="clean_corrected_eval_charts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = load_rows(root)
    avg_rows = average_rows(rows)
    plot_dir = root / "outputs" / "wandb_clean_eval_plots"
    plot_infos = save_clean_plots(rows, avg_rows, plot_dir)

    import wandb
    run = wandb.init(entity=args.entity, project=args.project, name=args.run_name, group="clean-corrected-eval", job_type="eval-visualization")

    metric_table = wandb.Table(columns=["group", "run", "split", "path"] + [label for _, label, _ in METRICS])
    for row in rows + avg_rows:
        metric_table.add_data(row["group"], row["run"], row["split"], row["path"], *[row[key] for key, _, _ in METRICS])

    payload = {"clean/eval_metrics_table": metric_table}
    for key, title, path in plot_infos:
        safe = title.replace("@", "at").replace("^", "pow").replace(" ", "_").replace("/", "_")
        payload[f"clean/charts/{safe}"] = wandb.Image(str(path))
    wandb.log(payload)
    run.finish()
    print("uploaded clean charts", args.entity, args.project, args.run_name)
    for _, _, p in plot_infos:
        print(p)

if __name__ == "__main__":
    main()
