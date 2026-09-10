# Sources and licenses

Project additions are MIT licensed; see [LICENSE](../LICENSE). Third-party material retains its own terms.

- [llama.cpp](https://github.com/ggml-org/llama.cpp): exact commit in `dependencies.json`; [MIT notice](../LICENSES/llama.cpp-MIT.txt). The compact-mask kernel is adapted from our archived private fork. The Qwen3.5 baseline uses stock attention arithmetic; the new patch omits the earlier accumulation-precision and tile-tuning experiments.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): architecture, configuration and chat template; Apache 2.0. [License text](../LICENSES/Qwen-Apache-2.0.txt).
- [Unsloth Qwen3.5-4B GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF): the downloaded Q4_K_M quantization; exact revision and file checksum pinned. No vision projector or multimodal evaluation is included.
- [MoBA paper](https://arxiv.org/html/2502.13189v1): the block-selection approach, not an originality claim for this repository.
- [FrogNano report](https://arxiv.org/html/2609.07925v2): motivation for evaluating compact coding agents. We have not obtained its trained checkpoint, reproduced its training, or reproduced its benchmark scores.
- Included source passages retain their [input attribution](../data/README.md). The three function tasks were authored for this pilot and are not taken from SWE-bench.

The archived project's licenses and original attribution remain applicable to its contents. Existing cached model weights and raw artifacts are local and ignored, not distributed through Git.
