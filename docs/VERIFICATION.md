# Verification of the fresh Qwen3.5 project

- The stock benchmark records the SHA-256 of an empty upstream diff. Its model checkpoint hash was verified before execution.
- Both full-mask and compact-mask builds completed on the pinned backend. All 20 inference executions completed: one stock baseline, two five-trial calibration loops, three 4K validation passes, and six coding completions.
- All five matched full-mask/compact-mask trials have byte-identical prompt-token, sampled-score and continuation-token files. The dense controls also match the stock baseline on these observed outputs.
- All three freely generated coding token sequences match between dense and compact MoBA. Both modes pass 2/3 tasks and fail the same two overlap cases.
- The final six-file upstream patch applies to clean pinned source files, reproduces local source changes exactly, and passes reverse-application checks. The setup command rebuilds the native runner and verifies the model checksum.
- Driver syntax and active documentation links were checked. No unit-test suite or full-model training loop was added.
- Earlier result/source files remain under `archive/qwen3-4b/`. Their old paths and recorded metrics retain their historical meaning; their setup should be run from a separate restored workspace.

The early search summaries selected against their first dense control. After observing timing variation, the current autoresearch driver was tightened to require at least 3% improvement over both controls before promoting a sparse candidate. This is an exploratory selection margin, not statistical significance. The original recorded loop scripts are preserved beside their results; no historical measurement was overwritten. Neither recorded 2K search clears the new speed margin. No new GPU run was needed to assess this reporting criterion.

## 32K extension

A late-start Vulkan profiler was added and exercised on a 2K sparse replay before the long runs. Its timed-pass sampled NLL and continuation metrics exactly matched the previous compact run. The 32K dense and MoBA runs each completed prefill, 64 decode steps, alignment padding, and a diagnostic 256-token append batch. All 512 sampled positions and target IDs matched, and the input token files matched byte-for-byte. Main prefill/decode timings exclude padding and the instrumented batch.

The new runner records a fresh protocol and keeps the top-32 policy fixed. Profile aggregation is reproducible with `experiments/profile_summary.py`. The profiler is a single-device diagnostic; operator serialization and its one append batch limit interpretation.
