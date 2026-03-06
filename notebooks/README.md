# 交互式文档与实验

> 这里同时包含两类内容：适合直接在线阅读的 HTML 讲解稿，以及适合在本地 / Colab 中运行的 notebook 源文件。为了更适合 GitBook 阅读，本页按“先看什么最值”重新排了顺序。

## 推荐先读的 HTML 文档

- [多头注意力分化机制](multi-head-divergence.html)
- [注意力机制演进与推理全景](attention-evolution-and-inference.html)
- [MHA vs MLA](mha-vs-mla-full-derivation.html)
- [MHA vs GQA](mha-vs-gqa-full-derivation.html)
- [MHA vs DSA](mha-vs-dsa-full-derivation.html)
- [MHA vs 线性注意力](mha-vs-linear-attention-full-derivation.html)
- [统一对比表](attention-mechanisms-unified-comparison.html)

## Notebook 源文件

- [Python / NN / PyTorch 基础实战](python_nn_pytorch_fundamentals_workshop.ipynb)
- [Mini Transformer from Scratch](mini_transformer_from_scratch_workshop.ipynb)
- [Attention / Tokenizer / Beam Search](attention_tokenizer_beamsearch.ipynb)
- [LLM 推理基础](llm_inference_fundamentals.ipynb)
- [PagedAttention 与 LRU](kv_cache_paged_lru_workshop.ipynb)
- [量化精度实验](quantization_precision_experiment.ipynb)
- [投机解码模拟器](speculative_decoding_simulator.ipynb)
- [分布式 Roofline 分析](distributed_inference_roofline.ipynb)
- [PPO / GRPO 实现](rl_ppo_grpo_implementation.ipynb)
- [GRPO 训练流程图](grpo_training.png)
- [推理模型 Workshop](reasoning_models_workshop.ipynb)
- [RAG Prefix Caching 模拟](rag_prefix_caching_simulator.ipynb)
- [vLLM 架构走读](vllm_architecture_walkthrough.ipynb)
- [MLA 潜在空间分析](mla_latent_space_analysis.ipynb)
- [大海捞针评测](needle_in_haystack_demo.ipynb)
- [LLM 系统相关算法题](leetcode_llm_system_related.ipynb)

## GitBook 阅读建议

- HTML 导出页更适合直接在线阅读，公式和版式也更稳定。
- `.ipynb` 更适合在本地 Jupyter 或 Colab 中运行和修改。
- 如果你想把 notebook 里的结论和仓库实现对上，先看 [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md)。
- From Scratch 相关实验入口见 [../src/from_scratch/README.md](../src/from_scratch/README.md)。
