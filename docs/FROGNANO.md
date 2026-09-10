# FrogNano reproduction assessment

Checked September 10, 2026. The [authors’ project page](https://microsoft.github.io/debug-gym/) links the technical report without a FrogNano checkpoint or a dedicated Leaf/TaskPilot release. A public Hugging Face model search for `FrogNano` returned no models. The [paper page](https://huggingface.co/papers/2609.07925) has no linked model. This is an availability check, not a statement that a release will never happen.

[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) is available and is the starting checkpoint identified by the paper. Our pinned Unsloth Q4_K_M GGUF is a quantized derivative, not FrogNano.

## What the report specifies

FrogNano alternates five rounds of policy-adaptive synthetic-task generation and reinforcement learning. Leaf exposes read, write, edit, glob and bash. Tasks are validated using reference patches and hidden grading tests. DPPO trains with groups of eight trajectories, 256 trajectories per update, and 200 updates per round. Its implementation uses eight NVIDIA B200 GPUs: two for training and six for asynchronous rollouts. Context budgets reach approximately 131K. The efficiency reward penalizes long successful generations. These details come from [the report, sections 2–4 and Appendix A.1](https://arxiv.org/html/2609.07925v2).

## What we can reproduce here

We can implement a small independent harness, executable task validation, a measured curriculum, and local Qwen3.5 inference experiments. These would be method replications at a different scale. Matching the published scores requires the trained checkpoint and matching evaluation conditions, or an independently trained model with comparable resources and artifacts. Neither has been established here.

The existing Vulkan workers from the Qwen3 project trained small attention-router proxies. They do not implement full Qwen3.5 backpropagation, optimizer state, asynchronous agent rollouts, or DPPO. They cannot be presented as FrogNano training.

The first new experiment measures full attention against selective MoBA on the available Qwen3.5 checkpoint. A small coding smoke evaluation checks whether the inference path can still solve simple executable tasks. It is not SWE-bench or a repository-level agent evaluation. Before claiming a practical improvement, extend to held-out repositories, multiple task attempts, cached multi-turn contexts, and solve rate per unit wall time.
