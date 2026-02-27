# Python 核心语法回顾与进阶

> 面向 LLM 开发的 Python 核心技能速查，从函数式编程到 NumPy/Pandas 高频操作

---

## 学习建议（配套实战）

- 配套 Notebook：`notebooks/python_nn_pytorch_fundamentals_workshop.ipynb`
- 建议顺序：先看本篇的语法速查，再跑 Notebook 的 Part A，把每段代码改写一遍。
- 建议目标：能独立写出流式数据读取、数据清洗模板、计时装饰器和并发调用骨架。

---

## 一、函数式编程

### 1.1 高阶函数

```python
# map / filter / reduce
nums = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, nums))          # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, nums))   # [2, 4]

from functools import reduce
total = reduce(lambda a, b: a + b, nums)            # 15
```

### 1.2 装饰器 (Decorator)

```python
import time
from functools import wraps

def timer(func):
    """计时装饰器 —— 训练/推理性能分析必备"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_step(batch):
    ...
```

### 1.3 生成器 (Generator)

```python
def data_stream(file_path, chunk_size=1024):
    """大文件流式读取 —— 预训练数据加载常用"""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# 惰性求值，内存友好
for chunk in data_stream("train.jsonl"):
    process(chunk)
```

### 1.4 列表/字典推导式

```python
# 快速构建 vocab
vocab = {token: idx for idx, token in enumerate(token_list)}

# 嵌套推导
pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
```

---

## 二、面向对象编程 (OOP)

### 2.1 类与继承 —— PyTorch 模型定义核心

```python
import torch.nn as nn

class BaseAttention(nn.Module):
    """所有注意力变体的基类"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        raise NotImplementedError

class MultiHeadAttention(BaseAttention):
    """标准 MHA 继承基类"""
    def __init__(self, d_model, n_heads):
        super().__init__(d_model, n_heads)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
```

### 2.2 魔术方法

```python
class TokenizedDataset:
    """自定义数据集，支持 len/索引/迭代"""
    def __init__(self, texts, tokenizer):
        self.encodings = [tokenizer.encode(t) for t in texts]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

    def __repr__(self):
        return f"TokenizedDataset(n={len(self)})"
```

### 2.3 dataclass —— 轻量配置对象

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    model_name: str = "qwen2.5-7b"
    lr: float = 2e-5
    batch_size: int = 4
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
```

---

## 三、NumPy 核心操作

### 3.1 张量创建与变形

```python
import numpy as np

# 常用创建
x = np.random.randn(4, 128, 64)   # (batch, seq_len, d_head)
mask = np.triu(np.ones((128, 128)), k=1)  # 上三角 causal mask

# 变形操作（对应 PyTorch 的 view/reshape/transpose）
x_t = x.transpose(0, 2, 1)       # (4, 64, 128) —— 转置
x_r = x.reshape(4, -1)           # (4, 8192)   —— 展平
```

### 3.2 广播机制 (Broadcasting)

```python
# 注意力分数 + 因果 mask
scores = np.random.randn(4, 8, 128, 128)  # (B, H, L, L)
mask = np.triu(np.ones((1, 1, 128, 128)), k=1) * -1e9  # 广播到 (B, H, L, L)
masked_scores = scores + mask
```

### 3.3 矩阵运算

```python
Q = np.random.randn(128, 64)  # (L, d_h)
K = np.random.randn(128, 64)

# 注意力分数
scores = Q @ K.T / np.sqrt(64)  # (L, L)

# 爱因斯坦求和 —— 灵活的张量收缩
# 多头注意力一步写完
attn_out = np.einsum('bhld,bhld->bhl', Q_multi, K_multi)
```

---

## 四、Pandas 基础 —— 数据分析与数据集处理

### 4.1 数据加载与探索

```python
import pandas as pd

# 加载训练数据
df = pd.read_json("train.jsonl", lines=True)

# 快速探索
print(df.shape)               # (50000, 3)
print(df.columns.tolist())    # ['instruction', 'input', 'output']
print(df['output'].str.len().describe())  # 输出长度统计
```

### 4.2 数据清洗

```python
# 去重
df = df.drop_duplicates(subset=['instruction'])

# 过滤太短/太长的样本
df = df[df['output'].str.len().between(10, 4096)]

# 缺失值处理
df['input'] = df['input'].fillna("")
```

### 4.3 数据集构建

```python
# 构造 SFT 训练数据
def build_prompt(row):
    return f"<|im_start|>user\n{row['instruction']}\n{row['input']}<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"

df['text'] = df.apply(build_prompt, axis=1)

# 按 token 长度分桶统计
df['token_len'] = df['text'].apply(lambda x: len(x.split()))
print(df['token_len'].value_counts(bins=10))

# 导出
df[['text']].to_json("sft_data.jsonl", orient='records', lines=True)
```

---

## 五、Python 并发 —— LLM 服务端必备

### 5.1 asyncio 异步 I/O

```python
import asyncio
import aiohttp

async def call_llm(session, prompt):
    """异步调用 LLM API"""
    async with session.post(
        "http://localhost:8000/v1/completions",
        json={"prompt": prompt, "max_tokens": 512}
    ) as resp:
        return await resp.json()

async def batch_inference(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = [call_llm(session, p) for p in prompts]
        return await asyncio.gather(*tasks)

# 并发 100 个请求
results = asyncio.run(batch_inference(prompts[:100]))
```

### 5.2 多进程 —— 数据预处理加速

```python
from multiprocessing import Pool

def tokenize_chunk(texts):
    return [tokenizer.encode(t) for t in texts]

# 8 核并行 tokenize
with Pool(8) as p:
    chunks = [texts[i::8] for i in range(8)]
    results = p.map(tokenize_chunk, chunks)
```

---

## 六、LLM 开发高频 Python 技巧

| 技巧 | 用途 | 示例 |
|------|------|------|
| `@contextmanager` | 资源管理（GPU 显存） | `with torch.no_grad():` |
| `typing` 类型注解 | 代码可读性 | `def forward(x: Tensor) -> Tensor:` |
| `json / jsonlines` | 训练数据 I/O | `jsonl` 格式读写 |
| `tqdm` | 进度条 | `for batch in tqdm(dataloader):` |
| `logging` | 训练日志 | 替代 print，支持级别和文件输出 |
| `pathlib.Path` | 路径操作 | `Path("checkpoints") / f"step_{step}"` |
| `os.environ` | 环境变量 | `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"` |

---

## 面试一句话
- "Python 的装饰器和生成器是训练框架的核心模式，NumPy 的广播机制映射到 PyTorch 张量操作，asyncio 是高并发 LLM 服务的基础。"
