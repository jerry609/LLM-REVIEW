# From Scratch 模块

> 这一层用于统一承载 tokenizer、模型、训练、推理、对齐和具体 recipe 的从零复现实现。

## 结构说明

- `common/`：公共工具与共享组件。
- `tokenizer/`：分词器与词表实验。
- `model/`：骨干网络与层级模块。
- `data/`：数据处理与数据集封装。
- `training/`：预训练、SFT、优化器、训练循环。
- `inference/`：生成、采样、缓存与推理工具。
- `alignment/`：PPO、GRPO、DPO 等对齐训练相关逻辑。
- `recipes/`：按专题组织的复现路线，如 S1、GRPO、DAPO、MoE、多模态。

## 当前状态

- 目录骨架已经创建完成。
- 具体实现会优先按公共组件优先、recipe 后补的方式推进。