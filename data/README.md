# Inputs for the Qwen3.5 pilot

The text passages are reused, unchanged, from the archived Qwen3 experiment. They are not new independent corpora or guaranteed absent from model pretraining.

- `code.txt`: a frozen copy of pinned llama.cpp `ggml/src/ggml-quants.c`, under its MIT license. Used for the initial Qwen3.5 calibration and as unrelated context in the synthetic coding smoke evaluation.
- `technical.txt`: a frozen copy of pinned llama.cpp `tools/server/README.md`, under the same MIT license.
- `shakespeare.txt`: the text from [char-rnn tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), authored by William Shakespeare. Preserved for future language comparisons; no new Qwen3.5 Shakespeare result is claimed.
- `coding.json`: three newly authored function tasks with fixed cases. The cases are used for grading, not included in the model prompt. This small, public task set does not establish repository-level coding ability.

Benchmark records include prompt checksums and actual input token IDs. Long-text prefill truncates a single passage to the requested length; it does not repeat text to fill the context. Coding prompts retain the entire request and use the official non-thinking assistant prefix.
