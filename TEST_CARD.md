# Qwen3.5-4B test card

**Status: 1.52x scored-prefill throughput at 32K on one passage; no meaningful improvement at 2K/4K. Approximate attention, no FrogNano reproduction or model training.**

Model: Unsloth Qwen3.5-4B Q4_K_M GGUF, with exact revision and SHA-256 in [dependencies.json](dependencies.json). Hardware: Intel Core i5-13500H, Iris Xe RPL-P integrated GPU, 32 GiB system RAM, Linux/Vulkan. The model log confirms 33/33 model/output layers offloaded to Vulkan. FP16 KV, batch 256, four CPU threads; submission cap 8 and serialized submissions. [Hardware record](results/hardware.json), [stock GPU log](results/stock/dense-2k.log).

## Progress and measurements

The stock backend was restored to the pinned upstream commit before the first run. It took 19.720 s for 2K scored prefill and decoded at 5.41 tokens/s. Later dense controls reproduced its sampled score files and continuation tokens exactly. All five corresponding full-mask/compact-mask runs also produced byte-identical sampled-score, prompt-token and continuation-token files. This checks the observed values, not every hidden activation or vocabulary logit.

| Stage | Policy | Scored prefill, s | Decode, tokens/s | Sampled PPL change | Continuation agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| full-mask-2k | dense | 18.828 | 6.02 | +0.00% | 32/32 |
| full-mask-2k | top8-tail2 | 19.071 | 6.02 | +1.42% | 31/32 |
| full-mask-2k | top16-tail2 | 19.247 | 5.92 | -0.98% | 32/32 |
| full-mask-2k | top16-tail4 | 19.139 | 5.97 | -0.84% | 32/32 |
| full-mask-2k | dense-repeat | 18.931 | 5.84 | +0.00% | 32/32 |
| compact-2k | dense | 19.954 | 5.66 | +0.00% | 32/32 |
| compact-2k | top8-tail2 | 18.783 | 5.85 | +1.42% | 31/32 |
| compact-2k | top16-tail2 | 19.044 | 5.99 | -0.98% | 32/32 |
| compact-2k | top16-tail4 | 19.026 | 5.61 | -0.84% | 32/32 |
| compact-2k | dense-repeat | 19.004 | 6.05 | +0.00% | 32/32 |
| validation-4k | dense | 40.984 | 5.71 | +0.00% | 64/64 |
| validation-4k | moba | 40.904 | 5.56 | +1.02% | 64/64 |
| validation-4k | dense-repeat | 41.018 | 5.61 | +0.00% | 64/64 |

The expanded-mask implementation did not beat its dense controls at 2K. Compact top-8 appeared 6.2% faster than the first dense pass, but only 1.2% faster than the dense repeat. This control variation prevents a strong speed claim. The 4K top-32 policy was fixed before evaluating the technical passage; it retained the same terminal block fraction as top-16 at 2K. Its 0.2% timing advantage is effectively a tie, while sampled perplexity increased 1.02%.

[Full-mask calibration](results/full-mask-2k/results.json), [compact calibration](results/compact-2k/results.json), [frozen 4K validation](results/validation-4k/results.json).

## Interpretation

The architecture has eight full-attention layers and 24 linear-attention layers. Only eligible full-attention prefill layers are modified. The eight-layer count is not a measurement of attention's runtime share; operator profiling is still needed. Decode uses the original attention algorithm but its KV history can differ after sparse prefill. There is no demonstrated decode improvement.

Prefill sums synchronized model calls with sampled logits enabled. Model loading and CPU loss reduction are excluded from that sum; `prefill_wall_s` includes the loss reduction. The 2K workload samples 32 positions; 4K samples 64. Sampling is every 32 tokens in the second half of a non-repeated passage. Continuations are teacher-forced along 32 or 64 dense-generated tokens. These are not ordinary-prefill timings, free-generation agreement, statistical-equivalence tests, or a diverse loss benchmark.

All three 2K sparse candidates passed the exploratory PPL-ratio <=1.02 and continuation-agreement >=0.95 checks. These are calibration checks on the selection input, without a confidence-bound requirement. The 4K result's point estimates also pass, but neither establishes losslessness or general coding quality. No Qwen3.5 8K result is available. The later 32K comparison is reported separately below.

## 32K long-context comparison

**Measured 1.52x scored-prefill throughput, with +0.74% sampled perplexity on one code passage.** Two timed passes were run, with no dense timing repeat or policy tuning.

| Measurement | Dense | Compact MoBA |
| --- | ---: | ---: |
| Scored prefill | 745.92 s (12.43 min) | 491.99 s (8.20 min) |
| Decode | 4.53 tokens/s | 4.53 tokens/s |
| Sampled NLL | 0.158058 | 0.165450 |
| Sampled perplexity | 1.171235 | 1.179924 |
| Continuation agreement with dense | 64/64 | 64/64 |

Prefill time fell 34.04%. Natural-target argmax agreement was 508/512. Continuation NLL changed by -0.000276 nats/token. The paired position-bootstrap 95% interval for the sampled perplexity ratio was [0.999345, 1.016361]. That interval resamples positions; it does not account for sequence correlation or establish general quality.

Top-32 was kept fixed from the 4K validation: 32 of 512 blocks at the end of 32K, with two final full-attention layers dense. This is much more aggressive than 32 of 64 blocks at 4K. Decode used dense attention. The source was one previously used, non-repeated code passage, with 512 sampled targets and up to 64 teacher-forced continuation tokens. This is not a 32K coding-task benchmark, ordinary-prefill-only timing, or evidence of lossless attention.

### Why the benefit appears at long context

The timed prefill/decode ran with profiling disabled. Afterward, alignment padding and an additional 256-token batch were processed, enabling Vulkan timestamps only for the final diagnostic batch (starting at token 33,024). These instrumented timings serialize operators and describe that append batch, not the entire prefill.

| Diagnostic GPU operator group | Dense | MoBA |
| --- | ---: | ---: |
| Flash-attention kernels | 7.532 s (77.9%) | 2.770 s (54.8%) |
| DeltaNet core | 0.629 s (6.5%) | 0.603 s (11.9%) |
| Matrix multiplies | 1.230 s (12.7%) | 1.251 s (24.8%) |
| Pooling/top-k/scatter | 0.002 s (0.0%) | 0.034 s (0.7%) |
| Other kernels | 0.276 s (2.9%) | 0.393 s (7.8%) |

Flash-attention kernels became the dominant cost in the dense diagnostic batch despite only eight layers using full attention. Matrix multiplies include projections and feed-forward computation; the DeltaNet row covers its core operator, not all work in its layers. Routing overhead spans several groups. The profile supports targeting long-context full attention; it does not imply all contexts or tasks benefit equally.

[Raw comparison](results/validation-32k/results.json), [protocol](results/validation-32k/protocol.json), [dense profile](results/validation-32k/dense.profile.json), [MoBA profile](results/validation-32k/moba.profile.json).

## Coding smoke evaluation

| Mode | Solved | Total elapsed, s | Solves/hour on these three tasks |
| --- | ---: | ---: | ---: |
| dense | 2/3 | 96.49 | 74.62 |
| moba | 2/3 | 96.80 | 74.38 |

Both modes passed clamp and signed-duration tasks, and both missed empty-interval cases in the overlap task. All three generated token sequences matched across modes. These are freely generated completions, unlike the teacher-forced continuation comparison above. Each mode was run once per task; there is no statistically established throughput difference.

Prompts contained about 2.8K tokens of unrelated source context plus a short function request. Generation used the official non-thinking prefix and a 160-token cap. Grading ran five fixed cases per function using a restricted Python grammar. End-to-end elapsed time includes a fresh model load for each task. This is a synthetic single-turn smoke evaluation, not a coding-agent or SWE-bench result. The solves/hour figure is merely an extrapolation of these three task timings.

[Task cases](data/coding.json), [coding records](results/coding-smoke/results.json).

## Reproduction

See [README commands](README.md#run-experiments). The standalone backend is rebuilt with `python3 scripts/setup.py --model`; `--stock` reconstructs the unmodified upstream baseline. The final patch applies to clean pinned source files, matches the local source changes, and reverses cleanly. Current Python drivers require no third-party packages.

The earlier Qwen3-4B results, including its 1.99x 32K measurement, are [archived separately](archive/qwen3-4b/TEST_CARD.md). They do not describe this model. [FrogNano feasibility](docs/FROGNANO.md) records what can and cannot currently be reproduced.
