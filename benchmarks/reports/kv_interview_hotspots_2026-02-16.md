# KV面试高频考点抓取分析（2026-02-16）

## 抓取范围
- 成功抓取：15/15 个来源
- 来源类型：框架官方文档（vLLM/HF/TensorRT-LLM/SGLang）、学术论文（arXiv/ACL）、面试向文章

## 高频考点（按覆盖来源数排序）
| 排名 | 主题 | 覆盖来源数 | 关键词命中数 |
|---|---|---:|---:|
| 1 | 显存估算与容量规划 | 15 | 341 |
| 2 | KV缓存原理(为什么加速) | 14 | 481 |
| 3 | PagedAttention与块管理 | 14 | 364 |
| 4 | 性能指标(TTFT/TPOT/吞吐/延迟) | 14 | 205 |
| 5 | KV压缩/量化 | 13 | 320 |
| 6 | Prefill/Decode拆分 | 10 | 226 |
| 7 | KV驱逐策略 | 10 | 158 |
| 8 | 前缀缓存/复用 | 8 | 128 |
| 9 | 分层缓存与Offload | 6 | 163 |
| 10 | 连续批处理/调度 | 6 | 97 |
| 11 | 复杂度与收益(O(n^2)->O(n)) | 3 | 20 |

## 结论（用于出题）
- Top1: 显存估算与容量规划
- Top2: KV缓存原理(为什么加速)
- Top3: PagedAttention与块管理
- Top4: 性能指标(TTFT/TPOT/吞吐/延迟)
- Top5: KV压缩/量化
- Top6: Prefill/Decode拆分

## 来源清单
- [OK] HF Transformers KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- [OK] vLLM Prefix Caching docs: https://docs.vllm.ai/en/stable/design/prefix_caching.html
- [OK] vLLM Quantized KV Cache docs: https://docs.vllm.ai/usage/quantization/quantized_kvcache/
- [OK] TensorRT-LLM KV Cache System docs: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
- [OK] NVIDIA KV Cache priority eviction blog: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/
- [OK] SGLang HiCache design: https://docs.sglang.ai/advanced_features/hicache_design.html
- [OK] H2O paper: https://arxiv.org/abs/2306.14048
- [OK] SnapKV paper: https://arxiv.org/abs/2404.14469
- [OK] PyramidInfer paper: https://aclanthology.org/2024.findings-acl.195/
- [OK] KeepKV paper: https://arxiv.org/abs/2504.09936
- [OK] KVzip paper: https://arxiv.org/abs/2505.23416
- [OK] LoRC paper: https://arxiv.org/abs/2410.03111
- [OK] Aliyun interview article: https://developer.aliyun.com/article/1704743
- [OK] 53AI interview article: https://www.53ai.com/news/AImianshi/2025073132590.html
- [OK] Cnblogs KV cache article: https://www.cnblogs.com/rossiXYZ/p/18811723
