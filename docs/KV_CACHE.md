# Reducing the Qwen3.5 KV cache

This experiment separates two ideas inspired by [DeepSeek-V4.1's technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf): storing fewer bits per cache entry and sharing cache banks between layers. It does not convert Qwen into DeepSeek's causal encoder-decoder architecture.

## Cache precision

`BENCH_KV=f16`, `q8_0`, or `q4_0` selects an existing llama.cpp cache format. Both K and V use the chosen format. The quantized paths use the pinned backend's Hadamard rotations. Q4_0 is blockwise integer quantization, not DeepSeek's FP4 format. The model's Q4_K_M weight quantization remains unchanged.

Q8_0 stores 32 values in 34 bytes and Q4_0 stores them in 18 bytes, including scales. Their cache reductions relative to FP16 are therefore 1.882x and 3.556x. These are KV-cache ratios, not total system-memory ratios. Model weights, recurrent state, compute buffers, and allocation padding must be reported separately.

The first 2K code smoke checks both retained 32/32 teacher-forced continuation agreement. The subsequent 4K technical-passage comparison sampled 256 natural targets. Four-bit MoBA used 40.5 MiB of KV instead of 144 MiB, retained 64/64 continuation agreement, and had a sampled PPL ratio of 0.99599 against dense FP16. Its prefill was slightly slower at this context length. [Pilot measurements](../results/kv-pilot/results.json).

The 32K experiment runs a fresh dense FP16 control, the existing MoBA FP16 policy, and MoBA Q4_0. All use the same context allocation, top-32 policy, two final dense full-attention layers, 512 sampled targets, and 64 teacher-forced continuation tokens. Profiling and tensor capture are disabled during timing. [Protocol](../results/kv-32k/protocol.json), [completed measurements](../results/kv-32k/results.json).

The four-bit run measured 535.91 seconds of scored prefill versus 745.17 seconds dense: **1.390x throughput**. KV fell from 1,040 to 292.5 MiB, **3.556x smaller**. Sampled PPL increased **1.195%**, natural argmax agreement was 506/512, and continuation agreement was 64/64. Decode measured 4.544 versus 4.531 tokens/s. The GPU workspace grew from 247.50 MiB with FP16 MoBA to 367.52 MiB with Q4_0, leaving a KV-plus-workspace saving of 627.48 MiB against FP16 MoBA. This meets the exploratory storage/speed/quality target on the measured passages.

```sh
python3 experiments/kv_cache.py --out runs/kv-4k
python3 experiments/kv_cache.py --out runs/kv-32k --tokens 32768 --prompt data/code.txt --stride 32 --types f16 q4_0
```

The quality gates are sampled PPL ratio <= 1.02 and continuation agreement >= 0.95. Passing is an exploratory check, not evidence of losslessness or statistical equivalence. Both passages were used in earlier experiments; this is not an untouched benchmark suite.

## A shared-cache adaptation pilot

The optional `BENCH_SHARE_MASK` assigns one bit to each full-attention pair: layer 7 reads layer 3's KV, layer 15 reads layer 11's, layer 23 reads layer 19's, and layer 31 reads layer 27's. Layer numbers are zero-based. The receiving layer does not allocate or write its own KV bank. Its own queries, gate, output projection, and surrounding DeltaNet/FFN layers remain. This changes prefill and decode attention, unlike the original prefill-only MoBA policy. Sharing is disabled by default.

The initial training experiment enables only bit 0, isolating layer 7. It first captures the original and shared layer's attention outputs on 2,048 Shakespeare tokens. A 256-by-256 affine map for each of 16 query heads predicts the original output before the existing attention gate. The adapter has 1,052,672 parameters; the original model weights remain frozen.

The fit uses the first 1,536 tokens. Vulkan computes the centered covariance matrices. NumPy solves the small regularized normal equations on CPU and selects among four ridge strengths using the remaining 512 tokens. This is hybrid Vulkan/CPU closed-form distillation, not full-model GPU backpropagation. The fit objective is attention-output squared error, not language-model loss. A separate 4K technical-passage run measures whether the fitted adapter actually preserves model quality.

```sh
python3 -m venv .venv-kv
.venv-kv/bin/pip install numpy==2.5.3
.venv-kv/bin/python experiments/share_kv.py --out results/my-sharing-pilot --work runs/my-sharing-captures
```

The driver builds the small Vulkan covariance executable, captures teacher/student outputs, fits an adapter, and compares dense, unadapted sharing, and adapted sharing. Raw captures and covariance arrays belong under ignored `runs/`; the fitted `adapter.weights`, its checksum, training metadata, and scored inference results are retained. `BENCH_ADAPTER` loads this pilot adapter for layer 7 only. One shared pair reduces eight banks to seven, not to four. Any move to four shared banks needs its own training and quality validation.

Shrinking the cache with existing quantization is an engineering result. Demonstrating that a small adapter can make independently trained Qwen layers share KV would be a separate research result. Neither should be described as a reproduction of DeepSeek's trained architecture.

## Sharing result: the simple adapter did not preserve quality

The Vulkan covariance calculation completed successfully, and the deployed adapter matched its fitted transform within 0.0314% relative RMSE. Calibration selection MSE improved by 78.9%. However, on the separate technical passage, the fitted adapter increased sampled PPL by 38.59% and retained only 52/64 continuation predictions. Unadapted sharing had a sampled PPL change of -1.61% but retained only 59/64 predictions, also below the gate. Natural argmax agreement was 190/256 with the adapter and 215/256 without it.

Both shared configurations physically allocated seven banks (126 MiB versus 144 MiB dense), and the adapter additionally allocated 4,210,688 bytes on Vulkan. No meaningful prefill speed improvement appeared. This rejects the output-only affine fit, not cross-layer sharing in general. Improving it would require an objective or adapter that better preserves the actual attention computation and downstream predictions; lower local output MSE alone was insufficient here.

The architectural path remains opt-in and is not the selected 32K configuration. [Training records](../results/kv-sharing/training.json), [quality comparison](../results/kv-sharing/results.json). The successful result is the existing MoBA policy combined with Q4_0 KV, with the measured quality tradeoff stated above.
