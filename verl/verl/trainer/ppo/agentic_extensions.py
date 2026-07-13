"""Agentic GRPO scheduling hooks used by the ToolAgent-GRPO fork."""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in TRUE_VALUES


def integer(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool
    split_json: str
    schedule: str
    seed: int

    @classmethod
    def from_env(cls) -> "CurriculumConfig":
        return cls(
            enabled=flag("GRPO_SEEN_CURRICULUM_ENABLED", False),
            split_json=os.getenv("GRPO_SEEN_CURRICULUM_SPLIT_JSON", "experiments/sft_collect_airline/split.json"),
            schedule=os.getenv("GRPO_SEEN_CURRICULUM", "40:0.85,100:0.60,300:0.40"),
            seed=integer("GRPO_SEEN_CURRICULUM_SEED", 20260701),
        )

    def ratio(self, step: int) -> float:
        result = 1.0
        for item in self.schedule.split(","):
            if not item.strip():
                continue
            until, value = item.split(":", 1)
            if step <= int(until):
                return float(value)
            result = float(value)
        return result


class CurriculumSampler:
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.covered: set[int] = set()
        self.uncovered: set[int] = set()
        path = Path(config.split_json)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.covered = {int(x) for x in data.get("covered_seen_task_ids", [])}
            self.uncovered = {int(x) for x in data.get("uncovered_seen_task_ids", [])}

    def choose(self, available: Iterable[int], count: int, step: int) -> tuple[list[int], dict[str, float]]:
        available = [int(x) for x in available]
        covered = [x for x in available if x in self.covered]
        uncovered = [x for x in available if x in self.uncovered]
        target = self.config.ratio(step)
        if not self.config.enabled or not covered or not uncovered:
            chosen = [available[i % len(available)] for i in range(count)] if available else []
        else:
            rng = random.Random(self.config.seed + step)
            chosen = []
            for _ in range(count):
                source = covered if rng.random() < target else uncovered
                if not source:
                    source = uncovered if source is covered else covered
                chosen.append(source[rng.randrange(len(source))])
        return chosen, {
            "curriculum/covered_count": float(sum(x in self.covered for x in chosen)),
            "curriculum/uncovered_count": float(sum(x in self.uncovered for x in chosen)),
            "curriculum/covered_ratio_target": float(target),
        }


@dataclass(frozen=True)
class BNDSPPlan:
    selected_rollouts: int
    skipped_easy: int
    root_top_p_sum: float


def plan_b_ndsr(scores: list[float]) -> BNDSPPlan:
    minimum = integer("B_NDSR_ROOT_MIN_SAMPLES", 4)
    # The original B-NDSR run used 4 initial root rollouts and allowed the
    # budget to grow through 6, 8, 10, up to 12 selected rollouts.
    maximum = integer("B_NDSR_ROOT_MAX_SAMPLES", 12)
    increment = integer("B_NDSR_ROOT_INCREMENT", 2)
    top_ps = tuple(float(x) for x in os.getenv(
        "B_NDSR_ROOT_TOP_PS",
        "0.85,0.90,0.92,0.95,0.88,0.96,0.90,0.97,0.85,0.90,0.92,0.95",
    ).split(","))
    labels = [float(x) >= 0.5 for x in scores]
    if labels and all(labels):
        count, skipped = minimum, 1
    elif len(set(labels)) > 1:
        count, skipped = min(maximum, minimum + increment), 0
    else:
        count, skipped = maximum, 0
    # Make a short user-supplied schedule safe for a larger rollout budget by
    # cycling it rather than silently under-counting root_top_p_sum.
    root_top_p_sum = sum(top_ps[i % len(top_ps)] for i in range(count)) if top_ps else 0.0
    return BNDSPPlan(count, skipped, root_top_p_sum)


def b_ndsr_metrics(plans: list[BNDSPPlan]) -> dict[str, float]:
    if not plans:
        return {
            "b_ndsr/selected_rollouts": 0.0,
            "b_ndsr/skipped_easy_tasks": 0.0,
            "b_ndsr/root_top_p_sum": 0.0,
        }
    return {
        "b_ndsr/selected_rollouts": sum(x.selected_rollouts for x in plans) / len(plans),
        "b_ndsr/skipped_easy_tasks": float(sum(x.skipped_easy for x in plans)),
        "b_ndsr/root_top_p_sum": sum(x.root_top_p_sum for x in plans) / len(plans),
    }


def _toolagent_training_path() -> None:
    root = os.getenv("TOOLAGENT_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)


def apply_judge_reward(reward_tensor, batch) -> tuple[object, dict[str, float]]:
    """Apply outcome-anchored Judge scores when the ToolAgent module is available."""
    if not flag("LLM_JUDGE_ENABLED", False):
        return reward_tensor, {}
    _toolagent_training_path()
    try:
        from toolagent.training.llm_judge import JudgeConfig, densify_trajectory
    except Exception:
        return reward_tensor, {"judge/enabled": 0.0}
    config = JudgeConfig.from_env()
    messages = batch.non_tensor_batch.get("llm_judge_messages")
    task_ids = batch.non_tensor_batch.get("task_id", batch.non_tensor_batch.get("uid", []))
    if messages is None:
        return reward_tensor, {"judge/enabled": 1.0, "judge/missing_messages": 1.0}
    rewards = reward_tensor.clone()
    # Judge is only meaningful as a relative ranking signal inside a task
    # group that already contains a ground-truth successful trajectory.  Do
    # not densify an all-failure task group: there is no success anchor for
    # calibrating its failed rollouts.
    task_keys = [str(task_ids[idx]) if idx < len(task_ids) else str(idx) for idx in range(len(messages))]
    success_by_task: dict[str, bool] = {}
    for idx, task_key in enumerate(task_keys):
        success_by_task[task_key] = success_by_task.get(task_key, False) or float(rewards[idx].sum().item()) >= 1.0
    judged = 0
    skipped_without_success = 0
    for idx, raw in enumerate(messages):
        if not isinstance(raw, list):
            continue
        old = float(rewards[idx].sum().item())
        if old < 1.0 and not success_by_task.get(task_keys[idx], False):
            skipped_without_success += 1
            continue
        dense = densify_trajectory(old, raw, str(task_ids[idx]), config)
        if old < 1.0:
            response_mask = batch.batch["response_mask"][idx]
            valid = response_mask.nonzero(as_tuple=False).flatten()
            if len(valid):
                rewards[idx].zero_()
                rewards[idx, valid[-1]] = dense
                judged += 1
    return rewards, {
        "judge/enabled": 1.0,
        "judge/judged_failures": float(judged),
        "judge/skipped_without_success_anchor": float(skipped_without_success),
        "judge/mean_reward": float(rewards.sum(-1).mean().item()),
    }


def jass_metrics(batch) -> dict[str, float]:
    if not flag("JASS_ENABLED", False):
        return {}
    _toolagent_training_path()
    try:
        from toolagent.training.jass import select_checkpoint
    except Exception:
        return {"jass/enabled": 0.0}
    traces = batch.non_tensor_batch.get("b_ndsr_trace", [])
    selected = 0
    for trace in traces:
        if trace and select_checkpoint([trace]) is not None:
            selected += 1
    return {"jass/enabled": 1.0, "jass/selected_checkpoints": float(selected)}
