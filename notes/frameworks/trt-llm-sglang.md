# TensorRT-LLM & SGLang

## TensorRT-LLM (NVIDIA)
### 特点
- NVIDIA 官方推理框架，针对 NVIDIA GPU 深度优化
- 基于 TensorRT 的图优化 + 自定义 CUDA kernel
- 支持 FP8/INT4 量化、in-flight batching、paged attention

### 与 vLLM 对比
| 维度 | vLLM | TRT-LLM |
|------|------|---------|
| 生态 | 开源社区 | NVIDIA 官方 |
| 部署 | pip install | 编译构建（较复杂） |
| 性能 | 高 | 更高（专用 kernel） |
| 灵活性 | 高（PyTorch 模型直接加载） | 需要转换模型格式 |
| MoE 支持 | 好 | 好 |

### 使用场景
- 追求极致性能 + 使用 NVIDIA GPU → TRT-LLM
- 快速原型 + 多平台 → vLLM

## SGLang
### 特点
- 结构化生成（JSON、正则表达式约束）原生支持
- RadixAttention：高效前缀缓存
- 自动并行 + 批处理

### RadixAttention
- 用 Radix Tree 管理 KV Cache 前缀
- 多请求共享相同前缀时自动复用
- 粒度比 vLLM 更细（token 级匹配）

### 约束解码
```python
@sgl.function
def gen_json(s):
    s += "Generate a JSON:" + sgl.gen("output", regex=r'\{"name": "\w+", "age": \d+\}')
```
- 在采样时强制输出符合给定约束
- 用 FSM (有限状态机) 或正则引擎做 token mask

## 面试一句话
- "TRT-LLM 适合生产部署追求极致性能；SGLang 的 RadixAttention 和约束解码对 agentic 场景特别有用。"
