# ToolAgent-GRPO Technical Report

## 1. Overview

ToolAgent-GRPO studies reinforcement learning for long-horizon tool-using agents in the tau-bench airline environment. The task requires an assistant to interact with a user simulator, call airline tools, track environment state, and eventually complete a realistic user request.

The project builds a complete post-training loop:

```text
72B teacher rollout
        -> trajectory filtering
        -> Qwen2.5-7B LoRA SFT
        -> veRL GRPO training
        -> multi-sample policy evaluation
```

The main focus is improving GRPO when terminal rewards are sparse and most early rollout groups fail completely.

## 2. Task Setting

The environment is tau-bench airline. Each task contains a hidden user goal and environment state. The policy must ask or answer the user, call tools, inspect tool results, and decide when the task is complete.

A trajectory contains:

- user simulator messages
- assistant responses
- tool calls
- tool results
- environment transitions
- terminal outcome reward

The project wraps 14 airline tools and supports multi-turn interaction with long context.

## 3. SFT Stage

The SFT stage trains Qwen2.5-7B with LoRA on filtered teacher trajectories. The dataset applies loss only to assistant output tokens. User messages, system context, and tool results are masked out with `IGNORE_INDEX`.

This stage gives the policy an initial ability to:

- follow airline task instructions
- emit valid assistant messages
- call domain tools
- maintain multi-turn context
- complete simple tool-use trajectories

## 4. GRPO Stage

The RL stage uses veRL GRPO with a 7B policy model and a 72B-AWQ user simulator. The training loop performs asynchronous rollout, tool execution, reward collection, and policy optimization.

Key engineering choices include:

- LoRA policy training
- FlashAttention
- gradient checkpointing
- FSDP parameter and optimizer offload
- logprob micro-batching
- vLLM GPU memory control
- separated policy and user simulator servers
- 24K context rollout support

These choices reduce memory pressure and make long multi-turn rollout more stable.

## 5. Curriculum and Judge Reward

### 5.1 Why Ordinary GRPO Is Hard Here

In long-horizon airline tasks, terminal rewards are sparse. Early in training, many rollout groups are all failures. If all samples in a GRPO group receive the same reward, the relative advantage signal becomes weak or uninformative.

This makes it hard to distinguish:

- completely wrong trajectories
- trajectories that queried useful information but missed one constraint
- trajectories that reached a near-success state but failed the final operation

### 5.2 Seen-Task Curriculum Sampling

The seen-task curriculum adjusts the training task distribution so that the model receives more useful learning signals during early and middle training. It gradually increases exposure to harder or less-covered tasks after the policy becomes more competent.

The goal is to improve:

- reward diversity inside rollout groups
- useful partial-success trajectories
- training stability under sparse rewards
- exploration efficiency on long-horizon tasks

### 5.3 Outcome-Anchored LLM-as-Judge

The outcome-anchored LLM-as-Judge mechanism keeps successful trajectories anchored by the real environment reward. For failed trajectories, it provides finer-grained process scores so that GRPO can rank failures by how close they are to success.

This gives the policy more informative relative advantages without replacing the ground-truth terminal reward for successful episodes.

## 6. Evaluation Protocol

Evaluation uses 50 tau-bench airline tasks with 4 samples per task.

The corrected metrics used in this report are:

- **pass@1**: only `sample_id=0` is counted for each task. A task contributes 1 if this first sample succeeds and 0 otherwise.
- **pass@4**: a task contributes 1 if any of its 4 samples succeeds.
- **pass^4**: a task contributes 1 only if all 4 samples succeed.
- **Avg. tool calls**: total tool calls divided by the number of samples.
- **Avg. turns**: total turns divided by the number of samples.
- **Error rate**: number of samples with non-empty error divided by the number of samples.

The report compares three curriculum runs against three ordinary GRPO runs.

## 7. Main Result

### 7.1 Three-Run Average

| Method | Runs | Avg. pass@1 | Avg. pass@4 | Avg. pass^4 | Avg. tool calls | Avg. turns | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary GRPO | 3 | 0.173 | 0.320 | 0.060 | 8.673 | 12.372 | 0.040 |
| Curriculum + Judge GRPO | 3 | **0.280** | **0.413** | **0.113** | **5.780** | **8.372** | **0.033** |

The curriculum + judge method improves average task success and reduces rollout cost:

- pass@1: 0.173 -> 0.280 (+10.7 points)
- pass@4: 0.320 -> 0.413 (+9.3 points)
- pass^4: 0.060 -> 0.113 (+5.3 points)
- average tool calls per sample: 8.673 -> 5.780
- average turns per sample: 12.372 -> 8.372

### 7.2 Six Individual Runs

| Group | Run | pass@1 | pass@4 | pass^4 | Avg. tool calls | Avg. turns | Error rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Curriculum | previous | 0.260 | 0.380 | 0.100 | 5.670 | 8.545 | 0.080 |
| Curriculum | repeat | 0.260 | 0.420 | 0.100 | 6.000 | 8.560 | 0.010 |
| Curriculum | unified | **0.320** | **0.440** | **0.140** | 5.670 | 8.010 | 0.010 |
| Ordinary GRPO | original | 0.200 | 0.300 | 0.060 | 9.070 | 13.035 | 0.060 |
| Ordinary GRPO | previous/current | 0.120 | 0.360 | 0.040 | 8.205 | 11.730 | 0.040 |
| Ordinary GRPO | repeat | 0.200 | 0.300 | 0.080 | 8.745 | 12.350 | 0.020 |

The six JSON files used for this comparison are stored under:

```text
outputs/curriculum/
  20260703_0438_curriculum_previous_eval_report.json
  20260703_115619_curriculum_repeat_eval_report.json
  20260703_131536_curriculum_unified_eval_report.json

outputs/grpo/
  20260701_grpo_original_eval_report.json
  20260703_115619_grpo_previous_current_eval_report.json
  20260703_repeat_grpo_eval_report.json
```

## 8. Figures


The following figures are included from the local experiment artifacts. They are kept in `docs/assets/` so that the report can display the original evaluation screenshots and tables.

### Figure 1

<p><img src="./docs/assets/result_01.png" alt="Result figure 1" width="900"></p>

### Figure 2

<p><img src="./docs/assets/result_02.png" alt="Result figure 2" width="900"></p>

### Figure 3

<p><img src="./docs/assets/result_03.png" alt="Result figure 3" width="900"></p>

### Figure 4

<p><img src="./docs/assets/result_04.png" alt="Result figure 4" width="900"></p>

### Figure 5

<p><img src="./docs/assets/result_05.png" alt="Result figure 5" width="900"></p>

### Figure 6

<p><img src="./docs/assets/result_06.png" alt="Result figure 6" width="900"></p>

### Figure 7

<p><img src="./docs/assets/result_07.png" alt="Result figure 7" width="900"></p>

### Figure 8

<p><img src="./docs/assets/result_08.png" alt="Result figure 8" width="900"></p>

### Figure 9

<p><img src="./docs/assets/result_09.png" alt="Result figure 9" width="900"></p>

### Figure 10

<p><img src="./docs/assets/result_10.png" alt="Result figure 10" width="900"></p>

### Figure 11

<p><img src="./docs/assets/result_11.png" alt="Result figure 11" width="900"></p>

## 9. Analysis

### 9.1 Why Curriculum Helps

The curriculum improves the density of useful training signals. Uniform sampling can waste a large fraction of rollout budget on all-failed groups. Curriculum sampling makes the early distribution more learnable, then raises difficulty as the policy improves.

### 9.2 Why Judge Reward Helps

Terminal reward only tells whether the final task succeeded. It does not tell whether a failed trajectory was close to success. The outcome-anchored judge provides relative scores among failed trajectories, which helps GRPO assign more meaningful advantages.

### 9.3 Tool Efficiency

Across the three-run comparison, the curriculum + judge group reduces average tool calls from 8.673 to 5.780 per sample and average turns from 12.372 to 8.372 per sample. This suggests that the policy learns a more direct tool-use strategy rather than relying on repeated or redundant tool queries.

## 10. Implementation Map

Important implementation files:

- `src/toolagent/sft_dataset.py`: assistant-only SFT labeling.
- `src/toolagent/verl_integration/agent_loop.py`: rollout loop for agent interaction.
- `src/toolagent/verl_integration/interaction.py`: user, environment, and tool interaction.
- `src/toolagent/training/b_ndsr.py`: B-NDSR related reward and advantage processing.
- `src/toolagent/training/jass.py`: JASS sampling and scoring logic.
- `src/toolagent/training/llm_judge.py`: LLM-as-Judge process reward for failed trajectories.

## 11. Limitations and Future Work

The current experiments focus on tau-bench airline. Future work can extend the framework to:

- additional tau-bench domains
- adaptive rollout budget allocation
- better failure-prefix reuse
- stronger process reward models
- online curriculum scheduling based on group reward diversity

## 12. Summary

ToolAgent-GRPO combines teacher rollout, LoRA SFT, veRL GRPO, tau-bench async rollout, curriculum sampling, outcome-anchored judge reward, and long-context memory optimization.

Across three comparable runs, the curriculum + judge variant improves average pass@1 from 0.173 to 0.280, pass@4 from 0.320 to 0.413, and pass^4 from 0.060 to 0.113, while reducing average tool calls from 8.673 to 5.780 per sample.
