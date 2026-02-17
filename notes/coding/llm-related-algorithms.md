# LLM 相关算法编程题精选

## 一、经典数据结构题（高频）

### 1. LRU Cache (LeetCode 146)
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```
**面试延伸**：如何将 LRU 扩展为 KV Cache eviction？需要考虑 block 粒度。

### 2. LFU Cache (LeetCode 460)
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.min_freq = 0
        self.key_val = {}
        self.key_freq = {}
        self.freq_keys = defaultdict(OrderedDict)

    def _update(self, key):
        freq = self.key_freq[key]
        self.freq_keys[freq].pop(key)
        if not self.freq_keys[freq]:
            del self.freq_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.key_freq[key] = freq + 1
        self.freq_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_val:
            return -1
        self._update(key)
        return self.key_val[key]

    def put(self, key: int, value: int) -> None:
        if self.cap <= 0:
            return
        if key in self.key_val:
            self.key_val[key] = value
            self._update(key)
            return
        if len(self.key_val) >= self.cap:
            evict_key, _ = self.freq_keys[self.min_freq].popitem(last=False)
            del self.key_val[evict_key]
            del self.key_freq[evict_key]
        self.key_val[key] = value
        self.key_freq[key] = 1
        self.freq_keys[1][key] = None
        self.min_freq = 1
```

---

## 二、Sampling 相关

### 3. Top-K Sampling
```python
import numpy as np

def top_k_sampling(logits, k, temperature=1.0):
    logits = logits / temperature
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = np.exp(top_k_logits - np.max(top_k_logits))
    probs /= probs.sum()
    return np.random.choice(top_k_idx, p=probs)
```

### 4. Top-P (Nucleus) Sampling
```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    selected_idx = sorted_idx[:cutoff]
    selected_probs = probs[selected_idx]
    selected_probs /= selected_probs.sum()
    return np.random.choice(selected_idx, p=selected_probs)
```

### 5. Beam Search
```python
def beam_search(model_fn, start_token, beam_width, max_len, eos_token):
    beams = [(0.0, [start_token])]  # (log_prob, tokens)
    completed = []
    for _ in range(max_len):
        candidates = []
        for score, tokens in beams:
            if tokens[-1] == eos_token:
                completed.append((score, tokens))
                continue
            logits = model_fn(tokens)
            log_probs = log_softmax(logits)
            top_k_idx = np.argsort(-log_probs)[:beam_width]
            for idx in top_k_idx:
                candidates.append((score + log_probs[idx], tokens + [idx]))
        beams = sorted(candidates, key=lambda x: -x[0])[:beam_width]
        if not beams:
            break
    completed.extend(beams)
    return max(completed, key=lambda x: x[0] / len(x[1]))
```

---

## 三、Tokenizer 相关

### 6. BPE (Byte Pair Encoding) 核心算法
```python
def bpe_train(corpus, num_merges):
    vocab = {}
    for word, freq in corpus.items():
        vocab[tuple(word)] = freq
    merges = []
    for _ in range(num_merges):
        pairs = {}
        for tokens, freq in vocab.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        new_vocab = {}
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges
```

---

## 四、Attention 相关

### 7. Multi-Head Attention (手写)
```python
import numpy as np

def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[-1]
    d_head = d_model // n_heads
    def split(x):
        return x.reshape(-1, n_heads, d_head).transpose(1, 0, 2)
    q, k, v = split(Q), split(K), split(V)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)
    out = attn @ v
    out = out.transpose(1, 0, 2).reshape(-1, d_model)
    return out
```

### 8. RoPE (Rotary Position Embedding)
```python
def rope(x, positions, base=10000):
    d = x.shape[-1]
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
    angles = positions[:, None] * freqs[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return np.stack([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1).reshape(x.shape)
```

---

## 五、系统设计编程

### 9. 简化版 Continuous Batching Scheduler
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Request:
    id: int
    prompt_tokens: int
    max_new_tokens: int
    generated: int = 0

class Scheduler:
    def __init__(self, max_batch_tokens: int = 4096):
        self.max_batch_tokens = max_batch_tokens
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def add(self, req: Request):
        self.waiting.append(req)

    def step(self):
        finished = []
        for req in self.running:
            req.generated += 1
            if req.generated >= req.max_new_tokens:
                finished.append(req)
        for req in finished:
            self.running.remove(req)
        decode_tokens = len(self.running)
        budget = self.max_batch_tokens - decode_tokens
        while self.waiting and budget > 0:
            req = self.waiting[0]
            if req.prompt_tokens <= budget:
                self.waiting.pop(0)
                self.running.append(req)
                budget -= req.prompt_tokens
            else:
                break
        return finished
```

### 10. Paged KV Cache
```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated = {}  # seq_id -> [block_ids]

    def allocate(self, seq_id, num_tokens):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("OOM: not enough free blocks")
        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.allocated[seq_id] = blocks
        return blocks

    def free(self, seq_id):
        if seq_id in self.allocated:
            self.free_blocks.extend(self.allocated.pop(seq_id))
```

---

## 六、面试策略

### 高频题目优先级
1. **LRU Cache**（必会，延伸到 KV Cache eviction）
2. **Top-K/Top-P Sampling**（LLM 基础）
3. **BPE 算法**（tokenizer 原理）
4. **Multi-Head Attention**（核心组件）
5. **Beam Search**（decode 策略）
6. **Scheduler**（系统设计）

### 做题技巧
- 先写接口定义，再写实现
- 注意边界条件（空输入、溢出）
- 复杂度分析要说清楚
- 能手写 attention 是加分项
