# Qwen3.5-4B sparse attention on an Intel Vulkan GPU

A fresh experiment measuring whether selective Mixture of Block Attention can make Qwen3.5-4B more useful on an Intel Iris Xe integrated GPU. This uses the publicly available Qwen3.5 checkpoint. The original Qwen model weights remain frozen.

Qwen3.5-4B combines 24 Gated DeltaNet linear-attention layers with eight full-attention layers. Our MoBA path changes only selected full-attention layers during aligned prefill. Decode remains dense. Findings from the previous Qwen3-4B project do not establish performance or quality for this architecture.

The 32K comparison reached **1.52x scored-prefill throughput** (12.43 to 8.20 minutes), with **+0.74% sampled perplexity** and 64/64 teacher-forced continuation agreement. This is one code passage and one timed pass per mode. The 2K/4K results remain essentially tied. [32K measurements](results/validation-32k/results.json).

A later cache experiment reached **3.56x smaller KV at 32K** (1,040 to 292.5 MiB) while retaining **1.39x scored-prefill throughput**. Four-bit KV increased sampled perplexity by 1.20% versus dense FP16 and retained 64/64 continuation agreement. Decode measured 4.54 versus 4.53 tokens/s. These are limited single-passage measurements. [Cache experiment](docs/KV_CACHE.md), [32K cache results](results/kv-32k/results.json).

## Experiment progression

1. Reset the root Git repository and removed `contrib/`. Preserved the earlier project under [archive/qwen3-4b](archive/qwen3-4b/README.md), including failed experiments and local ignored captures.
2. Restored unmodified pinned llama.cpp source and downloaded/checksummed the Qwen3.5-4B Q4_K_M checkpoint. Wrote a fresh inference benchmark.
3. Recorded a stock 2K baseline: 19.72 seconds of scored prefill and 5.41 decode tokens/s. Model loading was excluded from these two timings.
4. Ran five full-mask calibration trials: two dense controls and three sparse policies. All sparse trials passed the small calibration gates but were slightly slower than dense. Dense sampled scores and continuation tokens exactly matched the stock baseline. [Full-mask records](results/full-mask-2k/results.json).
5. Adapted the previous compact-mask kernel to Qwen3.5, retaining stock attention accumulation arithmetic and default tile tuning. The new router retains first/current blocks, uses 64-token blocks and leaves a configurable number of final full-attention layers dense. This is an implementation adaptation, not a new attention architecture. Corresponding full-mask and compact-mask runs produced identical sampled scores and continuation tokens.
6. Froze a top-32 policy and evaluated a different passage at 4K: 40.90 s sparse versus 40.98–41.02 s dense, +1.02% sampled perplexity, and 64/64 continuation agreement. The timing difference is negligible. [4K records](results/validation-4k/results.json).

7. Ran three function-writing tasks with freely generated outputs. Dense and MoBA both solved 2/3, produced identical token sequences, and missed the same empty-interval edge case. Their elapsed times were nearly equal. [Coding records](results/coding-smoke/results.json).

8. Ran the unchanged top-32 policy at 32K. Its 1.52x measured throughput gain contrasts with negligible short-context benefits. A separately instrumented append batch near 33K showed flash-attention kernels consuming 77.9% of dense GPU operator time. The sampled quality tradeoff and profiling limits are recorded in the test card.

9. Compared FP16, eight-bit and four-bit KV at short contexts, then ran fresh dense-FP16, MoBA-FP16 and MoBA-Q4_0 passes at 32K. Quantization uses existing llama.cpp formats; it reduces storage in addition to MoBA's attention-work reduction. The four-bit candidate passed the exploratory quality gates, with a smaller prefill speedup than FP16 MoBA. Raw results include the larger workspace required by quantized routing.

10. Tested actual cross-layer KV sharing and fitted a 1.05M-parameter attention-output adapter using Vulkan covariance calculations and CPU ridge solves. Calibration output error fell 79%, but separate-passage sampled PPL rose 38.6% and continuation agreement fell to 52/64. Unadapted sharing also failed the agreement gate. Sharing remains disabled; its failed checkpoint and measurements are retained. [Sharing records](results/kv-sharing/results.json).

11. Added persistent exact-prefix checkpoints containing both attention KV and DeltaNet recurrent state. At 32K, three different questions had bit-identical restored logits and tokens. Warm cache hits reduced complete 32-token benchmark time from about 555 seconds to 10.1–10.4 seconds (53–55x); paying the first full prefill/save gave 2.89x across three questions. This benefit requires an identical cached prefix and does not accelerate a new document. The checkpoint uses 339.13 MiB of storage. [Prefix-cache results and reproduction](docs/PREFIX_CACHE.md).

Current measurements and limitations are collected in the [test card](TEST_CARD.md).

## Setup

Run from this repository root on Linux with a working Vulkan driver. Required tools: Git, CMake, a C++17 compiler, Vulkan development headers, `glslc`, and Python 3.12 or newer. On Debian/Ubuntu the usual packages are `build-essential cmake git libvulkan-dev glslc python3`. Inference experiment drivers use only the Python standard library. The optional sharing-adapter fit also uses NumPy; see [its setup instructions](docs/KV_CACHE.md).

```sh
python3 scripts/setup.py --model
```

The script fetches the exact llama.cpp commit, applies our private patch, builds `bin/bench`, and downloads and verifies the 2.74 GB checkpoint in `models/`. All revisions and the model checksum are in [dependencies.json](dependencies.json). The existing build cache may be reused; a new checkout builds from the pinned sources. The initial shader/backend build can take several minutes on the recorded CPU.

For an unmodified upstream baseline, run `python3 scripts/setup.py --stock --model`, then run a dense benchmark. Run `python3 scripts/setup.py` to restore and rebuild the MoBA patch before sparse experiments. Switching backends rebuilds affected Vulkan sources; the recorded stock baseline predates all new patches.

## Run experiments

Use fresh output paths; runners refuse existing results.

```sh
MOBA=0 python3 experiments/benchmark.py --out runs/dense-2k
MOBA=1 MOBA_TOPK=16 MOBA_DENSE_LAYERS=2 python3 experiments/benchmark.py --out runs/moba-2k --reference runs/dense-2k.tokens
python3 experiments/autoresearch.py --out runs/search-2k
python3 experiments/coding.py --out runs/coding
python3 experiments/validate.py runs/validation-4k
python3 experiments/benchmark_32k.py runs/validation-32k
python3 experiments/profile_summary.py runs/validation-32k
python3 experiments/kv_cache.py --out runs/kv-32k --tokens 32768 --prompt data/code.txt --stride 32 --types f16 q4_0
python3 experiments/prefix_cache.py --out results/my-prefix-32k --work runs/my-prefix-32k --tokens 32768
```

The search runs exactly five trials: dense, top-8 with two final full-attention layers dense, top-16 with two dense, top-16 with four dense, and a dense repeat. It selects a passing sparse configuration only if it is at least 3% faster than both dense controls; otherwise it retains the faster dense control. This is a practical exploratory margin, not a significance test. The recorded early searches used the first dense pass for selection; this rule was tightened after observing their control variation. Use `--tokens 4096` for a longer calibration. Each benchmark subprocess has a 300-second timeout; the timeout is not permission to interpret a partial run as a successful comparison.

`MOBA_COMPACT=0` uses the expanded token mask for an implementation comparison. Both representations select the same blocks. Compact masks are specific to single-sequence aligned 256-token batches, 64-token blocks, and this scalar Vulkan backend. FP16 and Q4_0 KV were exercised at 32K; Q8_0 was exercised at 2K/4K. Short/unaligned batches and decode use dense attention. Other GPUs and arbitrary masks are not validated. MoBA alone retains full KV storage. `BENCH_KV=q8_0` or `BENCH_KV=q4_0` enables the separate cache-quantization path. Cross-layer sharing is an independent opt-in experiment described in [the cache research notes](docs/KV_CACHE.md).

The 32K command runs dense and fixed top-32 MoBA on `code.txt`, with 512 sampled targets and up to 64 teacher-forced continuation tokens. It keeps two final full-attention layers dense and has a 2,400-second timeout per pass. This is more aggressive than top-32 at 4K: only 32 of 512 blocks are retained at the end. It does not tune the policy or repeat the dense timing.

The prefix-cache command creates one persistent checkpoint, then compares three different questions against full recomputation in fresh processes. It verifies all generated-step vocabulary logits and separately measures file integrity checks, save, restore, suffix processing, decode, and startup. The checkpoint includes both KV and recurrent state. See [prefix-cache measurement boundaries](docs/PREFIX_CACHE.md). This is a controlled exact-prefix benchmark, not a serving scheduler.

The optional `--profile` benchmark flag appends a diagnostic 256-token batch after all timed prefill/decode work and alignment padding. A private Vulkan hook enables operator timestamps only for that extra batch. Profiling serializes operators; these timings describe an append batch near the final context length, not a breakdown of the whole timed prefill. Diagnostic work is included in process elapsed time but excluded from reported prefill/decode time. This profiling mode is for the single Vulkan device used here.

The validation command freezes top-32 with two final full-attention layers dense, then runs dense, compact MoBA and dense again at 4K on `technical.txt`. It does not select a new policy from that passage.

The coding command compares dense and top-16 MoBA on three synthetic function-writing tasks with unrelated source context. It uses the checkpoint’s non-thinking assistant prefix, a 160-token output limit, fixed executable cases, and a restricted Python grammar. There are no repository tools, agent loop, model updates, or SWE-bench instances. Solve rate per hour on three simple tasks is a smoke measurement, not an agent capability benchmark.

## Measurement boundaries

Prefill time sums synchronized model calls. With sampling enabled it includes the additional logits computation; CPU loss reduction is outside that sum and inside `prefill_wall_s`. These are scored-prefill measurements, not ordinary-prefill-only timings. Sampling covers every 32nd position in the second half of a non-repeated source passage. The 2K comparison has only 32 samples.

Later benchmark trials force the same dense continuation tokens, reporting next-token agreement and continuation NLL. Decode tokens/s measures synchronized one-token model calls and excludes CPU token selection, output writing, and model loading. Coding trials generate freely and have different output lengths; their end-to-end elapsed time includes model loading and grading.

The calibration gates are sampled perplexity ratio <=1.02 and continuation agreement >=0.95. They do not establish statistical equivalence or general quality. Training adaptive routers and a repository-level coding evaluation remain separate research steps. No claim of a breakthrough follows from this pilot.

## Layout

- `src/`: inference benchmark, Qwen3.5 block router, optional shared-cache adapter, and Vulkan covariance calculation.
- `experiments/`: baseline, five-trial search, coding smoke, cache comparison, prefix-checkpoint comparison, and adapter-fitting runners.
- `patches/`: private llama.cpp/Vulkan changes.
- `data/`, `configs/`: inputs, synthetic task cases and pinned model configuration/template.
- `results/`: measured records, including failures.
- `docs/`: research scope and attribution.
- `archive/`: the separate, earlier Qwen3-4B project.
- `vendor/`, `build/`, `bin/`, `models/`, `runs/`: ignored dependencies, generated tools and fresh runs.

The implementation was developed with an AI coding assistant. See [sources and licenses](docs/SOURCES.md).
