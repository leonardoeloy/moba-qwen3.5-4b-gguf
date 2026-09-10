# Local Leaf-inspired agent pilot

This is an independent, small implementation of the lightweight tool interface described in the [FrogNano report](https://arxiv.org/html/2609.07925v2), section 2. It uses the available Qwen3.5-4B Q4_K_M model with our MoBA/Q4_0 cache configuration. It does not use FrogNano weights, perform RL, synthesize tasks online, or reproduce SWE-bench scores.

## Reproduce

Run the repository's normal setup first, then install the optional template dependency in a separate environment. Linux Bubblewrap and GNU timeout must be installed (`bubblewrap` and `coreutils` on Debian/Ubuntu).

```sh
python3 scripts/setup.py --model
python3 -m venv .venv-leaf
.venv-leaf/bin/pip install Jinja2==3.1.6
.venv-leaf/bin/python experiments/leaf.py --out results/my-leaf-smoke --work runs/my-leaf-smoke
```

Add `--turns 6 --concise-tools` to reproduce the follow-up configuration in a separate output/work directory. It changes both the turn limit and the system instruction, so it is not an isolated prompt ablation.

The driver uses the pinned Qwen chat template, greedy non-thinking generation, five tools (`read`, `write`, `edit`, `glob`, `bash`), a four-turn limit and at most 256 generated tokens per turn. A complete response without tool calls ends the episode. Malformed calls and exhausted generation/turn limits are reported separately. Tools execute in order, and their results are returned in the next conversation turn.

File tools are restricted to a disposable task directory. Bash runs under Bubblewrap with no network, a cleared environment, read-only system tools, and only the task repository writable alongside temporary storage. Command execution has CPU, memory, output-size and wall-time limits. Grading runs after interaction using a restricted arithmetic Python grammar. The reference solution and grading cases are not mounted in the agent's environment or included in its prompts. They are public in the released fixture for reproducibility, so they are not a private held-out benchmark.

The two handcrafted repair fixtures cover empty half-open interval intersections and signed durations. The unchanged starter must fail at least one case, and the reference must pass all cases, before a rollout is accepted. This validates the task mechanics; it does not assess broad repository-level coding ability. They resemble the earlier function-writing smoke tasks and must not be described as independent generalization data.

Conversation prefill is recomputed in a fresh inference process each turn. The persistent-prefix optimization is not connected to this harness yet. Contexts are short; this pilot cannot establish a MoBA speed advantage. It records prompts, completions, tool responses, termination reasons, final candidates, inference metrics, and task grades. There is no dense-control harness comparison, stochastic pass@k estimate, policy update or efficiency-reward optimization.

## Measured attempts

| Configuration | Passing final patches | Natural finishes | Total episode time |
| --- | ---: | ---: | ---: |
| Original instruction, 4 turns maximum | 1/2 | 0/2 | 187.99 s |
| Concise-tool instruction, 6 turns maximum | 1/2 | 1/2 | 174.86 s |

In both runs, the interval task spent its fourth turn explaining cases and reached the 256-token generation cap without making an edit. The signed-duration task used `edit` to make the correct fix in turn three. The original run exhausted its turn allowance after reading the edited file; the follow-up finished naturally in turn five and passed all seven grading cases. The reference passes and starter failures were checked for both fixtures before generation.

The follow-up therefore demonstrates one complete file-repair episode, not a solve-rate improvement. It changes both an instruction and the allowed turn count, reuses the same tasks, and has one greedy attempt per case. Its modest total-time difference is not a validated speedup. Non-thinking generation did not prevent verbose reasoning in ordinary answer text.

[Original records](../results/leaf-smoke/results.json), [follow-up records](../results/leaf-concise/results.json). Each retains the exact driver, prompts, tool traces, final candidate, and inference records. The initial sandbox preflight failed because a process-count cap was below the host's existing usage; it was corrected before any rollout. No failed inference or truncated tool call was counted as a successful repair.

## Next research step

A working tool episode is only the first prerequisite. A meaningful follow-up needs less synthetic repository repairs, multiple stochastic attempts, separate calibration and evaluation tasks, and a direct comparison of tool interfaces and prefix reuse on the same tasks. Policy-adaptive task selection requires mixed-success rollouts; two greedy attempts do not provide that estimate. Actual FrogNano-style DPPO training additionally requires a supported full-model training backend, which this Vulkan inference implementation does not provide.
