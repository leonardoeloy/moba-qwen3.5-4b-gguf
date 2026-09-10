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

## KV precision and sharing extension

The new 32K dense-FP16, MoBA-FP16 and MoBA-Q4_0 passes completed successfully with the same allocation size and scoring positions. The fresh dense prompt, sampled scores and continuation tokens match the earlier 32K dense run byte-for-byte. Actual allocated KV falls from 1,040 to 292.5 MiB; the Q4_0 workspace increase is reported separately. All 512 scored position/target pairs match across these runs.

The sharing pilot captures a teacher and a seven-bank student, computes covariances on Vulkan, fits four ridge candidates on CPU, and evaluates the selected output adapter. A short native inference check reproduces its fitted transformation with relative RMSE 0.000314. The dense 4K validation scores and continuation tokens match the precision pilot's dense control exactly; all 256 scored position/target pairs align in the shared and adapted runs. Both shared variants fail the continuation gate; the adapted variant also fails the PPL gate. Neither is enabled by default.

The current seven-file backend patch passes clean application, byte-for-byte source comparison, and reverse checks at the pinned revision. The benchmark and new library changes compile, and the covariance executable runs on Vulkan0. New Python drivers pass syntax checks; no unit-test suite was added. Actual source snapshots are retained beside the new experiment records. The sharing snapshot predates two additional assertions in the current driver that check eight teacher banks and seven student banks; the recorded runs satisfy those assertions. Raw activation/covariance arrays remain in ignored `runs/` while the failed 4,210,688-byte adapter is retained as an experiment artifact.

## Persistent prefix checkpoints

The standalone prefix runner compiled against the unchanged pinned private backend. Both the 2K smoke and 32K experiment completed all seven subprocesses: three full recomputations, three warm restores, and one restore after a best-effort file-page eviction request. All restored prompt-token and generated-token files match their corresponding recomputation. Each restore checked 7,946,240 vocabulary logits with zero bitwise differences and zero maximum absolute error. Saved-state checksums match the retained manifests.

The current source snapshot, executable and runtime library fingerprints match the 32K protocol, and the backend diff still matches the seven-file patch. Python syntax and prefix-documentation links passed checks. The actual driver used at each stage is retained; the 32K driver adds runtime-change checks and broader environment isolation to the smoke driver. The three-question complete-process amortization was independently calculated from the raw trial records and saved in `results/prefix-32k/analysis.json`.

Raw checkpoints and logit arrays remain excluded by Git ignore rules. No unit-test suite was added. Restoration is exact on these observed same-policy runs; dense-model quality, long generations, arbitrary prefixes, production cache routing and sustained serving throughput were not evaluated by this experiment.

## Leaf-style harness pilot

Both two-task configurations completed. All 17 inference subprocesses exited successfully; one generation in each configuration hit its explicit token budget and was reported as such. Tool calls parsed and executed without reported tool errors, including an isolated Bash invocation. Starter implementations failed grading and reference implementations passed before each attempt. Final patch grades and natural termination are reported separately. Both actual driver versions and all traces are retained; the current driver matches the follow-up snapshot. No model parameters changed and no RL training was run.
