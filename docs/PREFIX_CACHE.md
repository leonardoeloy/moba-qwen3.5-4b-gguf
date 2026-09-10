# Persistent exact-prefix checkpoints

This experiment reuses a processed token prefix across different questions on Qwen3.5-4B. It uses the existing MoBA top-32 policy, two final dense full-attention layers, and Q4_0 KV. Cross-layer sharing and the failed adapter remain disabled. The original model weights are unchanged.

The checkpoint contains both attention KV and Gated DeltaNet recurrent state. It is saved before the question is appended, using llama.cpp's sequence-state API, and the state file is flushed with `fsync`. Each restore happens in a new process with a newly initialized model/context. The native runner checks the restored token IDs and final prefix position before processing the suffix.

The Python driver checks the model checksum, records the runtime library and executable hashes, and validates a manifest covering the source prefix, token limit, cache formats, context capacity, batching, threads, and attention policy. It verifies checkpoint integrity before each restore. The source-prefix hash is conservative: changing even an unused tail of the source file invalidates this benchmark's cache identity. This is a controlled benchmark, not a general prefix-matching server or cache eviction manager.

## Reproduce

After the normal repository setup:

```sh
python3 scripts/setup.py --model
python3 experiments/prefix_cache.py --out results/my-prefix-smoke --work runs/my-prefix-smoke --tokens 2048
python3 experiments/prefix_cache.py --out results/my-prefix-32k --work runs/my-prefix-32k --tokens 32768
```

The driver builds `bin/prefix` from `src/prefix.cpp`. It refuses occupied output/work directories. Raw checkpoints and vocabulary-logit captures are written under ignored `runs/`; manifests, measurements, generated text/tokens, source snapshots, and protocols are retained under the selected results directory. The native executable is a benchmark primitive; use the Python driver for the manifest checks.

Each experiment runs three questions about the same C-code excerpt:

1. Recompute the entire prefix and answer question 1, saving a checkpoint before the question.
2. Restore the checkpoint and answer question 1 again, then repeat after requesting file-page-cache eviction.
3. Recompute and restore separately for questions 2 and 3.

The prefix length is an exact multiple of 256 tokens. Prefix and nonempty question suffix are tokenized as explicit segments, with the same batching in cold and restored executions. Suffixes contain at most 256 tokens. Up to 32 output tokens are generated greedily. Memory-only checkpoints do not provide last-prefix logits; the nonempty suffix generates the logits needed to start answering.

## Measurement boundaries

Prefix timing is ordinary prefill without sampled vocabulary projections. It is not directly interchangeable with the earlier scored-prefill measurements. First-token timing here means output logits are ready; no HTTP server or streaming transport is included.

"Cold" in the file names means full prefix recomputation, not cold disk caches. The initial model-checksum verification warms the model file's page cache for all subprocesses.

The report separates model/context initialization, prefix computation, durable state save, integrity validation, state restoration, suffix processing, and synchronized decode calls. It reports both model-ready component speedups and complete subprocess speedups. Complete subprocess times also include tokenization, instrumentation, and cleanup. The one-time model fingerprint and initial manifest construction are outside the per-request measurements.

The three-request amortized result charges the first request for full prefill and checkpoint save; only the next two distinct questions get cache hits. It uses model-work components, not total driver wall time. The separately timed save is subtracted from the first uncached reference, since an uncached service would not save a checkpoint.

Standard restore measurements have a warm filesystem page cache: integrity verification reads the file immediately before launching the native runner. An additional restore uses `POSIX_FADV_DONTNEED` after that verification. This is best-effort file-page eviction, not a guarantee of a cold physical SSD or storage-controller cache. File-backed persistence is demonstrated by restoration in independent processes.

Every restored generation is compared against its own recomputed reference at every vocabulary logit, with bitwise float comparison and maximum absolute error recorded. Generated tokens and complete prompt-token files must also match exactly. This establishes observed restoration fidelity under the same MoBA/quantization policy; it does not establish equivalence to dense FP16, answer correctness, or fidelity for arbitrary workloads.

## 2K smoke check

At 2K, top-32 retains all 32 blocks, so this stage checks quantized hybrid-state restoration without actual block dropping. The 32K stage exercises genuinely sparse prefill.

All four restores (three questions and one eviction variant) reproduced all checked logits and generated tokens bit-for-bit. Each checked 7,946,240 vocabulary logits over 32 generated steps. The checkpoint was 71,623,472 bytes (68.31 MiB), and save plus `fsync` took 0.083 seconds.

Warm, validated model-ready first-token time fell from 20.22-20.33 seconds to 0.86-0.89 seconds (22.7-23.4x). Including decode, the model-work improvement was 4.38-4.51x; complete subprocess improvement was 3.70-3.82x. Across three distinct questions, paying the first prefill/save reduced model-work time from 74.24 to 35.96 seconds (2.06x).

The best-effort eviction restore took 0.060 seconds versus 0.014 seconds for the first warm restore; suffix processing and decode dominated at this short context. [Smoke records](../results/prefix-smoke/results.json).

The later 32K experiment uses the same native code. Its driver additionally rejects a changed runtime binary/library or model file during the experiment and strips extra llama.cpp/GGML environment overrides. Each stage retains the driver actually used. [32K protocol](../results/prefix-32k/protocol.json).

## 32K measured result

Three distinct question suffixes were each run with full recomputation and checkpoint restoration. A fourth restore requested file-page eviction. All four restores reproduced 32 generated tokens and 7,946,240 vocabulary logits each, bit-for-bit: **31,784,960 checked logits, zero mismatches**. This compares against the same MoBA/Q4_0 configuration, not dense FP16.

| Measurement | Full recomputation | Validated warm cache hit | Improvement |
| --- | ---: | ---: | ---: |
| Model-ready first-token components | 547.37–548.24 s | 2.66–2.91 s | 188–206x |
| Model-work components, including 32-token generation | 553.68–554.50 s | 8.64–9.03 s | 61.3–64.1x |
| Complete fresh-process benchmark | 555.07–555.88 s | 10.07–10.44 s | 53.2–55.2x |
| Three distinct questions, including initial prefill/save, process times summed | 1,666.38 s | 577.00 s | **2.89x** |

Complete-process cache-hit times include integrity validation and the 32-token comparison instrumentation. The first uncached reference excludes its separately timed save. Initial experiment fingerprinting and manifest publication remain outside these per-request sums. Model-work amortization gives 2.90x; the complete-process figure is the more inclusive comparison.

The checkpoint is **355,599,152 bytes (339.13 MiB)**. Saving plus `fsync` took **0.334 s**. Warm state restoration took **0.078 s**; checkpoint/runtime validation took approximately 0.27 s separately. Best-effort file-page eviction increased state restoration to **0.307 s**, with **10.69 s** for validation plus the complete process. All cases still process the new suffix and run ordinary dense decode. These measurements do not demonstrate a faster decode kernel.

The first request receives no prefix-reuse benefit. For an equal-cost serial workload, the measured mean process times imply approximately **1.96x** speedup at a hypothetical 50% hit rate, or **8.57x** at 90%. These are projections, not measured serving throughput; they omit cache-capacity effects, contention, eviction and per-miss checkpoint saves. Reusing one document does not establish those hit rates on real traffic.

[Full measurements](../results/prefix-32k/results.json), [amortization and projections](../results/prefix-32k/analysis.json), [checkpoint manifest](../results/prefix-32k/checkpoint.json). The small 2K and 32K records are retained; raw state/logit archives stay ignored. Three questions on one code prefix and one run per case establish a working local primitive, not broad model-quality or production-serving guarantees.

## What transfers from the 1,000-GPU diagram

The useful local idea is prefix affinity: requests about the same document can reuse its processed state. This experiment implements the persistence primitive behind that idea. A serving layer could later match exact token prefixes, route matching requests to the existing state, and enforce a cache-size limit. Those scheduling and eviction policies are not implemented here.

The diagram's separate prefill/decode pools, RDMA transfers, and expert/tensor parallel groups address multi-GPU deployment. On this single integrated GPU, this experiment provides no evidence that splitting workers would improve throughput. The checkpoint also does not shrink the active KV allocation: it avoids recomputing it. Its file size is additional persistent storage per cached prefix.

An engram tier represents a separate model or retrieval mechanism; a saved inference-state file is not an engram model. The screenshot alone does not supply trained weights or enough architectural detail to reproduce that component. The measured improvement here comes from exact-prefix reuse with the existing model.
