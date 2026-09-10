# Qwen3.5-4B test card

**Status: 1.52x scored-prefill throughput at 32K on one passage; no meaningful improvement at 2K/4K. Later KV quantization reaches 3.56x smaller cache with 1.39x prefill throughput. Approximate attention; no FrogNano reproduction or full-model training.**

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

## KV precision at 32K

A fresh three-pass comparison kept the MoBA policy fixed and varied cache precision. All passes used 33,280 allocated cells, with no diagnostic profiling or capture. The dense sampled-score and continuation-token files match the earlier 32K control byte-for-byte.

| Configuration | KV allocation | GPU workspace | Scored prefill | Speedup vs dense | Decode, tokens/s | Sampled PPL change | Continuation agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FP16 | 1,040 MiB | 266.26 MiB | 745.17 s | 1.00x | 4.53 | baseline | 64/64 |
| MoBA FP16 | 1,040 MiB | 247.50 MiB | 491.44 s | 1.52x | 4.61 | +0.74% | 64/64 |
| MoBA Q4_0 | 292.5 MiB | 367.52 MiB | 535.91 s | 1.39x | 4.54 | +1.20% | 64/64 |

Four-bit MoBA reduced actual KV allocation by **3.56x (71.875%)**, preserving a **28.08% prefill-time reduction** versus dense FP16. It was 9.05% slower in prefill than MoBA FP16. Its sampled PPL was approximately 0.45% higher than MoBA FP16, in addition to the attention approximation. Natural next-token argmax agreement with dense was 506/512 for Q4_0 and 508/512 for FP16 MoBA.

The reported KV-plus-workspace saving is 646.24 MiB versus dense and 627.48 MiB versus FP16 MoBA. The 2,603.5 MiB model buffer and 50.25 MiB recurrent-state buffer are unchanged. These buffer allocations do not measure total process peak RAM. Cache compression is not a 3.56x reduction in total memory.

The Q4_0 candidate passes the point-estimate PPL-ratio <=1.02 and continuation-agreement >=0.95 gates on this passage. A post-run paired-position bootstrap gives a PPL-ratio interval of 1.00159 to 1.02298; its upper end exceeds the 2% gate. This resampling does not account for sequence correlation or generalization. [Analysis](results/kv-32k/analysis.json). One pass per configuration and 512 sampled targets do not establish statistical equivalence, losslessness, or general long-context retrieval quality. Q4_0 is an existing integer cache format, not DeepSeek FP4 or a newly trained attention architecture. [Full records](results/kv-32k/results.json), [4K precision pilot](results/kv-pilot/results.json), [reproduction and architectural scope](docs/KV_CACHE.md).

## Shared-KV adapter: rejected

A separate dense-attention experiment made layer 7 read layer 3's KV bank. This physically reduced eight banks to seven. It fitted a 1,052,672-parameter affine attention-output adapter on Shakespeare, selecting a ridge strength from four candidates using a later portion of the same passage. Vulkan computed covariance matrices; CPU solved and scored the fits. Original Qwen weights stayed frozen.

| 4K technical-passage configuration | KV allocation | Sampled PPL change | Natural argmax agreement | Continuation agreement | Quality gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Original dense | 144 MiB | baseline | 256/256 | 64/64 | control |
| One shared pair | 126 MiB | -1.61% | 215/256 | 59/64 | fail |
| Shared pair plus fitted adapter | 126 MiB | +38.59% | 190/256 | 52/64 | fail |

The adapter adds 4,210,688 bytes of Vulkan weight storage beyond the KV allocation. All three scored-prefill times were approximately 41.4 seconds. The fit reduced calibration selection MSE from 0.38872 to 0.08198 (78.9%), but that proxy improvement did not transfer to language-model quality. Its Vulkan application passed a relative-RMSE check against the fitted transform (0.0314%). The failure is not evidence that all forms of learned KV sharing fail; it rejects this output-only affine adaptation on this setup.

**Sharing and the adapter remain off by default.** They were not combined with the successful 32K Q4_0 result, and no four-bank or 32K shared-cache quality claim is made. The failed checkpoint is retained for reproducibility. [Training](results/kv-sharing/training.json), [inference results](results/kv-sharing/results.json), [reproduction](docs/KV_CACHE.md).

## Persistent prefix reuse at 32K

Qwen3.5-4B with the existing MoBA top-32/two-dense-layer policy and Q4_0 KV saved both attention and recurrent state after a 32,768-token code prefix. Each question was evaluated by recomputation and restoration in independent processes. No model weights changed; shared KV and its adapter were disabled.

| Measurement | Recompute | Validated warm restore |
| --- | ---: | ---: |
| Model-ready first-token components | 547.37–548.24 s | 2.66–2.91 s (188–206x) |
| Complete 32-token benchmark process | 555.07–555.88 s | 10.07–10.44 s (53.2–55.2x) |
| Three distinct questions, summed process time, initial prefill/save charged | 1,666.38 s | 577.00 s (2.89x) |

The checkpoint occupied **339.13 MiB**, with **0.334 s** save plus fsync and approximately **0.078 s** warm restoration. A best-effort file-page eviction variant restored in **0.307 s** and completed in **10.69 s** including validation. Standard restores follow an integrity read, so their filesystem cache is warm.

All four restored runs matched their own recomputed prompt tokens, 32 generated tokens, and all 7,946,240 checked vocabulary logits exactly: **31,784,960 logits, zero bitwise mismatches**. This establishes observed checkpoint fidelity relative to MoBA/Q4_0, not equivalence to dense FP16 or answer correctness. New prefixes still pay full prefill. Active KV allocation is unchanged; saved files consume additional storage. Initial experiment fingerprinting/manifest publication are excluded, and complete-process times include instrumentation. One prefix and three short questions do not establish production hit rates or statistical performance guarantees.

[Results](results/prefix-32k/results.json), [amortization](results/prefix-32k/analysis.json), [2K smoke](results/prefix-smoke/results.json), [setup, timing definitions, and diagram analysis](docs/PREFIX_CACHE.md).
