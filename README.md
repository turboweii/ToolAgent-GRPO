# ToolAgent-GRPO: Long-Horizon Multi-Tool Agent RL

ToolAgent-GRPO is a long-horizon reinforcement learning training system for tool-using agents on the tau-bench airline environment. The project builds a complete post-training pipeline from 72B teacher rollouts to Qwen2.5-7B LoRA SFT and veRL GRPO training, with asynchronous interaction between a 7B policy model and a 72B-AWQ user simulator.

## Highlights

- Built a full 72B teacher rollout -> trajectory filtering -> Qwen2.5-7B LoRA SFT -> veRL GRPO pipeline for tau-bench airline.
- Integrated tau-bench with veRL async rollout, including multi-turn dialogue, 14 airline tools, environment feedback, and terminal rewards.
- Supported long-context rollout with LoRA, FlashAttention, gradient checkpointing, FSDP offload, logprob micro-batching, and vLLM memory control.
- Introduced seen-task curriculum sampling and outcome-anchored LLM-as-Judge reward shaping to improve GRPO under sparse terminal rewards and all-failed rollout groups.

## Main Result

Evaluation is performed on 50 tau-bench airline tasks with 4 samples per task. The README reports the average over three comparable runs for each method.

Metric definitions:

- **pass@1**: success of `sample_id=0` for each task, averaged over tasks.
- **pass@4**: a task is successful if any of its 4 samples succeeds.
- **pass^4**: a task is successful only if all 4 samples succeed.
- **Avg. tool calls** and **Avg. turns** are averaged per sample.

| Method | Runs | Avg. pass@1 | Avg. pass@4 | Avg. pass^4 | Avg. tool calls | Avg. turns | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary GRPO | 3 | 0.173 | 0.320 | 0.060 | 8.673 | 12.372 | 0.040 |
| Curriculum + Judge GRPO | 3 | **0.280** | **0.413** | **0.113** | **5.780** | **8.372** | **0.033** |

Compared with ordinary GRPO, the curriculum + judge variant improves average pass@1 by **+10.7 points**, pass@4 by **+9.3 points**, and pass^4 by **+5.3 points**, while reducing average tool calls from **8.673** to **5.780** per sample.

## Method

Long-horizon tool-use RL differs from ordinary single-turn RLHF. In tau-bench airline, the policy must interact with a user simulator, call external tools, track state, and satisfy a realistic user request. Terminal 0/1 rewards are sparse, and early GRPO rollout groups often fail completely, producing weak advantage signals.

ToolAgent-GRPO addresses this with:

- **Seen-task curriculum sampling**: adjusts task distribution to increase reward diversity and gradually raise training difficulty.
- **Outcome-anchored LLM-as-Judge reward**: keeps successful trajectories anchored by true environment rewards while assigning finer-grained process scores to failed trajectories.
- **Long-context rollout engineering**: supports 24K context and 15 user/assistant turns under controlled GPU memory usage.
- **Multi-sample evaluation**: reports pass@1, pass@4, average turns, average tool calls, and error rate by task split.


## Result Figures

The detailed report includes all experiment screenshots. Two representative figures are shown below.

<p><img src="./docs/assets/result_04.png" alt="ToolAgent result figure 4" width="900"></p>

<p><img src="./docs/assets/result_10.png" alt="ToolAgent result figure 10" width="900"></p>

## Report

A more detailed technical report with additional figures is available here:

[REPORT.md](./REPORT.md)

## Key Files

```text
configs/
  train/sft/
  train/grpo/
  eval/

scripts/
  train/sft/
  train/grpo/
  eval/
  vllm_server/

src/toolagent/
  sft_dataset.py
  training/b_ndsr.py
  training/jass.py
  training/llm_judge.py
  verl_integration/agent_loop.py
  verl_integration/interaction.py
```

## Note

The SFT stage uses assistant-only token-level cross-entropy loss. Curriculum sampling, B-NDSR, JASS, and LLM-as-Judge are applied during the GRPO rollout and reward stage, not during SFT.
