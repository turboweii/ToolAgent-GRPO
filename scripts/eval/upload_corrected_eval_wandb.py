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
    ("ordinary_grpo", "ordinary previous_current", "outputs/grpo/20260703_115619_grpo_previous_current_eval_report.json"),
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


def metric_from_split(split_data, key):
    if key == "avg_tool_calls_per_sample":
        return float(split_data.get("avg_tool_calls_per_sample", split_data.get("avg_tool_calls", 0.0)))
    return float(split_data[key])


def load_rows(root: Path):
    rows = []
    for group, run, rel in RUNS:
        path = root / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        by_split = data.get("by_split", {})
        for split in SPLITS:
            split_data = by_split[split] if split != "overall" else by_split.get("overall", data)
            row = {"group": group, "run": run, "split": split, "path": rel}
            for key, _, _ in METRICS:
                row[key] = metric_from_split(split_data, key)
            rows.append(row)
    return rows


def average_rows(rows):
    avg_rows = []
    for group in ["ordinary_grpo", "curriculum"]:
        for split in SPLITS:
            items = [r for r in rows if r["group"] == group and r["split"] == split]
            row = {"group": group, "run": f"{group} average", "split": split, "path": "average over 3 runs"}
            for key, _, _ in METRICS:
                row[key] = mean(r[key] for r in items)
            avg_rows.append(row)
    return avg_rows


def save_plots(rows, avg_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    run_order = [r[1] for r in RUNS]
    colors = {
        "curriculum previous": "#4c78a8",
        "curriculum repeat": "#72b7b2",
        "curriculum unified": "#54a24b",
        "ordinary original": "#f58518",
        "ordinary previous_current": "#e45756",
        "ordinary repeat": "#b279a2",
    }

    for key, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = list(range(len(SPLITS)))
        width = 0.12
        offsets = [(-2.5+i)*width for i in range(len(run_order))]
        for offset, run in zip(offsets, run_order):
            vals = []
            for split in SPLITS:
                item = next(r for r in rows if r["run"] == run and r["split"] == split)
                vals.append(item[key])
            label = run.replace("curriculum ", "cur-").replace("ordinary ", "grpo-").replace("previous_current", "prev")
            ax.bar([i + offset for i in x], vals, width, label=label, color=colors[run])
        ax.set_xticks(x, SPLITS)
        if key != "avg_tool_calls_per_sample":
            ax.set_ylim(0, 1.0 if key in {"success_rate", "error_rate"} else 0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Corrected eval by split: {title}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        path = out_dir / f"corrected_{key.replace('^','pow').replace('@','at')}_by_split.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path)

    # Average comparison for headline metrics.
    for key, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = list(range(len(SPLITS)))
        width = 0.36
        ordinary = [next(r for r in avg_rows if r["group"] == "ordinary_grpo" and r["split"] == s)[key] for s in SPLITS]
        curriculum = [next(r for r in avg_rows if r["group"] == "curriculum" and r["split"] == s)[key] for s in SPLITS]
        ax.bar([i - width/2 for i in x], ordinary, width, label="Ordinary GRPO avg", color="#f58518")
        ax.bar([i + width/2 for i in x], curriculum, width, label="Curriculum avg", color="#54a24b")
        ax.set_xticks(x, SPLITS)
        if key != "avg_tool_calls_per_sample":
            ax.set_ylim(0, 1.0 if key in {"success_rate", "error_rate"} else 0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Corrected eval 3-run average: {title}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        path = out_dir / f"corrected_{key.replace('^','pow').replace('@','at')}_average_by_split.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--entity", default="weixuan021103-tsinghua-university")
    parser.add_argument("--project", default="ToolAgent-GRPO-Eval")
    parser.add_argument("--run-name", default="corrected_toolagent_eval_by_split")
    parser.add_argument("--group", default="corrected-eval")
    parser.add_argument("--job-type", default="eval-correction")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = load_rows(root)
    avg_rows = average_rows(rows)
    plot_dir = root / "outputs" / "wandb_corrected_eval_plots"
    plot_paths = save_plots(rows, avg_rows, plot_dir)

    import wandb
    table = wandb.Table(columns=["group", "run", "split", "path"] + [label for _, label, _ in METRICS])
    for row in rows + avg_rows:
        table.add_data(row["group"], row["run"], row["split"], row["path"], *[row[key] for key, _, _ in METRICS])

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        group=args.group,
        job_type=args.job_type,
        config={"definition": "pass@1=sample0, pass@4=any success among 4, pass^4=all 4 success", "splits": SPLITS},
    )

    log_data = {"corrected/eval_table_by_split": table}
    for path in plot_paths:
        log_data[f"corrected/plots/{path.stem}"] = wandb.Image(str(path))

    # Summary metrics for overall split averages.
    overall_ord = next(r for r in avg_rows if r["group"] == "ordinary_grpo" and r["split"] == "overall")
    overall_cur = next(r for r in avg_rows if r["group"] == "curriculum" and r["split"] == "overall")
    for key, label, _ in METRICS:
        safe = label.replace("@", "at").replace("^", "pow").replace(" ", "_").replace("/", "_")
        log_data[f"corrected/overall_average/ordinary/{safe}"] = overall_ord[key]
        log_data[f"corrected/overall_average/curriculum/{safe}"] = overall_cur[key]
        log_data[f"corrected/overall_average/delta/{safe}"] = overall_cur[key] - overall_ord[key]

    wandb.log(log_data)
    for k, v in log_data.items():
        if isinstance(v, (int, float)):
            run.summary[k] = v
    run.finish()
    print("uploaded", args.entity, args.project, args.run_name)
    for path in plot_paths:
        print(path)

if __name__ == "__main__":
    main()
