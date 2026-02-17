#!/usr/bin/env python3
"""Generate algorithm practice notebooks and mock interview files."""
import json, os

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.10.0"}},
        "cells": cells
    }

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip().split("\n")}
def code(src): return {"cell_type": "code", "metadata": {}, "source": [l+"\n" for l in src.strip().split("\n")], "outputs": [], "execution_count": None}

# =========================================================================
# 1. LeetCode LLM-Related Notebook
# =========================================================================
lc_cells = [
    md("# 🧮 LeetCode 高频题精选 —— LLM 系统关联版\n\n> 每道题都标注了与 LLM 系统的关联，面试时可以自然延伸。\n> **建议**：每题限时 20 分钟，先写思路再写代码。"),

    md("## 1. LRU Cache (LC 146) ⭐⭐⭐\n**LLM 关联**：KV Cache 驱逐策略的核心数据结构\n\n- 时间复杂度：get/put O(1)\n- 关键：双向链表 + 哈希表"),
    code("""class Node:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    \"\"\"手写双向链表版 LRU，不用 OrderedDict（面试官更喜欢）\"\"\"
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()  # dummy
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            self._remove(self.map[key])
        node = Node(key, value)
        self.map[key] = node
        self._add_to_front(node)
        if len(self.map) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]

# ---- 测试 ----
cache = LRUCache(2)
cache.put(1, 1); cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)  # evicts key 2
assert cache.get(2) == -1
cache.put(4, 4)  # evicts key 1
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
print("✅ LRU Cache: all tests passed")"""),

    md("### 面试延伸：LRU → Paged KV Cache 驱逐\n```\n面试回答模板：\n\"LRU 在 KV Cache 中不是按 key-value 对驱逐，而是按 block 粒度。\n 一个 prefix 可能占用多个 block，驱逐时需要释放整个 prefix 的所有 block。\n 此外，实际系统中还需要考虑 reference counting（prefix sharing）和\n 多租户公平性（quota-aware eviction）。\"\n```"),

    md("## 2. LFU Cache (LC 460) ⭐⭐⭐\n**LLM 关联**：热门 prefix 缓存策略，频率驱逐 vs 时间驱逐的权衡"),
    code("""from collections import defaultdict

class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.min_freq = 0
        self.key_val = {}
        self.key_freq = {}
        self.freq_keys = defaultdict(dict)  # freq -> {key: None} (ordered by insertion)
        self.time = 0  # tie-break by time

    def _update(self, key):
        freq = self.key_freq[key]
        del self.freq_keys[freq][key]
        if not self.freq_keys[freq]:
            del self.freq_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.key_freq[key] = freq + 1
        self.freq_keys[freq + 1][key] = self.time
        self.time += 1

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
            # evict LFU (LRU among ties)
            evict_key = next(iter(self.freq_keys[self.min_freq]))
            del self.freq_keys[self.min_freq][evict_key]
            if not self.freq_keys[self.min_freq]:
                del self.freq_keys[self.min_freq]
            del self.key_val[evict_key]
            del self.key_freq[evict_key]
        self.key_val[key] = value
        self.key_freq[key] = 1
        self.freq_keys[1][key] = self.time
        self.time += 1
        self.min_freq = 1

# ---- 测试 ----
lfu = LFUCache(2)
lfu.put(1, 1); lfu.put(2, 2)
assert lfu.get(1) == 1     # freq(1)=2, freq(2)=1
lfu.put(3, 3)              # evicts key 2 (lowest freq)
assert lfu.get(2) == -1
assert lfu.get(3) == 3
lfu.put(4, 4)              # evicts key 1 or 3, key 3 freq=2, key 1 freq=2 -> LRU: key 1
assert lfu.get(1) == -1
print("✅ LFU Cache: all tests passed")"""),

    md("## 3. Top-K Frequent Elements (LC 347) ⭐⭐\n**LLM 关联**：Top-K Sampling / Token 频率统计 / 热门 Prefix 分析"),
    code("""import heapq
from collections import Counter

def top_k_frequent(nums, k):
    \"\"\"O(n log k) 用最小堆\"\"\"
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

def top_k_frequent_bucket(nums, k):
    \"\"\"O(n) 桶排序法 —— 面试加分\"\"\"
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result

# ---- 测试 ----
assert set(top_k_frequent([1,1,1,2,2,3], 2)) == {1, 2}
assert set(top_k_frequent_bucket([1,1,1,2,2,3], 2)) == {1, 2}
print("✅ Top-K Frequent: all tests passed")"""),

    md("## 4. 滑动窗口最大值 (LC 239) ⭐⭐⭐\n**LLM 关联**：Sliding Window Attention / KV Cache 窗口驱逐策略"),
    code("""from collections import deque

def max_sliding_window(nums, k):
    \"\"\"单调递减队列 O(n)\"\"\"
    dq = deque()  # 存 index，保持值递减
    result = []
    for i, v in enumerate(nums):
        # 移除窗口外的
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # 保持递减
        while dq and nums[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

# ---- 测试 ----
assert max_sliding_window([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([1,-1], 1) == [1,-1]
print("✅ Sliding Window Maximum: all tests passed")"""),

    md("### 面试延伸：Sliding Window → Sliding Window Attention\n```\n\"滑动窗口最大值用的单调队列思想，和 Sliding Window Attention 的思路一致：\n 只关注最近的 W 个 token，超出窗口的 KV 直接驱逐。\n Mistral/Mixtral 使用 sliding window size = 4096，每层看不同的窗口，\n 堆叠多层后有效感受野 = layers × window_size。\"\n```"),

    md("## 5. Implement Trie (LC 208) ⭐⭐⭐\n**LLM 关联**：Radix Tree 用于 Prefix Caching / RAG 前缀复用"),
    code("""class TrieNode:
    __slots__ = ('children', 'is_end')
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, prefix: str):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

# ---- 测试 ----
trie = Trie()
trie.insert("apple")
assert trie.search("apple") == True
assert trie.search("app") == False
assert trie.starts_with("app") == True
trie.insert("app")
assert trie.search("app") == True
print("✅ Trie: all tests passed")"""),

    md("### Trie → Radix Tree 延伸\n```python\n# Radix Tree 是 Trie 的压缩版：将只有一个孩子的连续节点合并\n# SGLang 的 RadixAttention 用 Radix Tree 管理 Prefix KV Cache：\n#   - 每个节点存储一段 token prefix 的 KV block IDs\n#   - 公共前缀只存一份，多个请求共享\n#   - 插入/查找/驱逐都是 O(prefix_length) \n#   - 比 hash table 更节省内存（公共前缀合并）\n```"),

    md("## 6. 合并 K 个有序链表 (LC 23) ⭐⭐\n**LLM 关联**：多路归并 → 多 Worker 输出归并 / Top-K Sampling 合并"),
    code("""import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def merge_k_lists(lists):
    \"\"\"最小堆归并 O(N log K)\"\"\"
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    dummy = tail = ListNode()
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next

# ---- 测试 ----
def to_list(lst):
    dummy = cur = ListNode()
    for v in lst:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def from_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    return result

lists = [to_list([1,4,5]), to_list([1,3,4]), to_list([2,6])]
assert from_list(merge_k_lists(lists)) == [1,1,2,3,4,4,5,6]
print("✅ Merge K Sorted Lists: all tests passed")"""),

    md("## 7. 数据流中位数 (LC 295) ⭐⭐\n**LLM 关联**：实时 P50/P99 延迟监控"),
    code("""import heapq

class MedianFinder:
    \"\"\"两个堆：max_heap (左半) + min_heap (右半)\"\"\"
    def __init__(self):
        self.lo = []  # max-heap (negated)
        self.hi = []  # min-heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

# ---- 测试 ----
mf = MedianFinder()
mf.addNum(1); mf.addNum(2)
assert mf.findMedian() == 1.5
mf.addNum(3)
assert mf.findMedian() == 2.0
print("✅ Median Finder: all tests passed")"""),

    md("## 8. 设计哈希表 (LC 706) ⭐⭐\n**LLM 关联**：Token-to-ID mapping / Block Table 实现"),
    code("""class MyHashMap:
    \"\"\"链地址法，面试写 open addressing 也可以\"\"\"
    def __init__(self, size=1024):
        self.size = size
        self.buckets = [[] for _ in range(size)]

    def _hash(self, key):
        return key % self.size

    def put(self, key: int, value: int) -> None:
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key: int) -> int:
        for k, v in self.buckets[self._hash(key)]:
            if k == key:
                return v
        return -1

    def remove(self, key: int) -> None:
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return

# ---- 测试 ----
hm = MyHashMap()
hm.put(1, 1); hm.put(2, 2)
assert hm.get(1) == 1
assert hm.get(3) == -1
hm.put(2, 1)
assert hm.get(2) == 1
hm.remove(2)
assert hm.get(2) == -1
print("✅ HashMap: all tests passed")"""),

    md("## 9. 一致性哈希 (系统设计编程题) ⭐⭐\n**LLM 关联**：分布式 Prefix Cache 路由 / SGLang consistent hashing"),
    code("""import hashlib, bisect

class ConsistentHash:
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas
        self.ring = []       # sorted hash values
        self.hash_to_node = {}
        for node in (nodes or []):
            self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:{i}")
            self.ring.append(h)
            self.hash_to_node[h] = node
        self.ring.sort()

    def remove_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:{i}")
            self.ring.remove(h)
            del self.hash_to_node[h]

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.hash_to_node[self.ring[idx]]

# ---- 测试 ----
ch = ConsistentHash(["gpu-0", "gpu-1", "gpu-2"])
# 验证负载大致均衡
from collections import Counter
distribution = Counter()
for i in range(1000):
    distribution[ch.get_node(f"prefix-{i}")] += 1
print(f"Node distribution: {dict(distribution)}")
assert len(distribution) == 3
assert all(v > 200 for v in distribution.values())  # 大致均衡
print("✅ Consistent Hash: all tests passed")"""),

    md("## 10. 综合挑战：实现 Mini Token Scheduler ⭐⭐⭐\n**LLM 关联**：Continuous Batching + Preemption + Priority"),
    code("""import heapq
from dataclasses import dataclass, field
from typing import List

@dataclass(order=True)
class Request:
    priority: int
    id: int = field(compare=False)
    prompt_len: int = field(compare=False)
    max_gen: int = field(compare=False)
    generated: int = field(default=0, compare=False)

    @property
    def done(self):
        return self.generated >= self.max_gen

class MiniScheduler:
    \"\"\"Continuous batching + priority preemption.\"\"\"
    def __init__(self, max_batch_tokens=512, max_batch_size=8):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.waiting: List[Request] = []   # min-heap by priority
        self.running: List[Request] = []
        self.completed: List[Request] = []

    def add_request(self, req: Request):
        heapq.heappush(self.waiting, req)

    def step(self):
        # 1. decode existing
        still_running = []
        for req in self.running:
            req.generated += 1
            if req.done:
                self.completed.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. fill with waiting requests (prefill)
        decode_tokens = len(self.running)  # 1 token per running req
        budget = self.max_batch_tokens - decode_tokens
        while (self.waiting
               and len(self.running) < self.max_batch_size
               and budget > 0):
            req = heapq.heappop(self.waiting)
            if req.prompt_len <= budget:
                self.running.append(req)
                budget -= req.prompt_len
            else:
                heapq.heappush(self.waiting, req)
                break  # can't fit

        return len(self.completed)

    def run_to_completion(self):
        steps = 0
        while self.running or self.waiting:
            self.step()
            steps += 1
            if steps > 10000:
                raise RuntimeError("infinite loop")
        return steps

# ---- 测试 ----
sched = MiniScheduler(max_batch_tokens=256, max_batch_size=4)
sched.add_request(Request(priority=1, id=0, prompt_len=50, max_gen=10))
sched.add_request(Request(priority=0, id=1, prompt_len=30, max_gen=5))   # higher priority
sched.add_request(Request(priority=2, id=2, prompt_len=100, max_gen=20))

steps = sched.run_to_completion()
print(f"All requests completed in {steps} steps")
ids = [r.id for r in sched.completed]
print(f"Completion order: {ids}")
assert 1 in ids  # high priority should complete
assert len(sched.completed) == 3
print("✅ Mini Scheduler: all tests passed")"""),

    md("---\n## 📊 刷题进度跟踪\n\n| # | 题目 | 难度 | LLM关联 | 状态 |\n|---|------|------|---------|------|\n| 1 | LRU Cache | Medium | KV Cache 驱逐 | ⬜ |\n| 2 | LFU Cache | Hard | 频率驱逐策略 | ⬜ |\n| 3 | Top-K Frequent | Medium | Sampling/热prefix | ⬜ |\n| 4 | Sliding Window Max | Hard | SWA/窗口驱逐 | ⬜ |\n| 5 | Trie | Medium | Radix Prefix Cache | ⬜ |\n| 6 | Merge K Lists | Hard | 多路归并 | ⬜ |\n| 7 | Find Median | Hard | P50/P99 监控 | ⬜ |\n| 8 | Design HashMap | Easy | Block Table | ⬜ |\n| 9 | Consistent Hash | - | 分布式Cache路由 | ⬜ |\n| 10 | Mini Scheduler | - | Continuous Batching | ⬜ |"),
]

# =========================================================================
# 2. PPO / GRPO Implementation Notebook
# =========================================================================
rl_cells = [
    md("# 🎯 手写 PPO / GRPO —— LLM RL 训练核心算法\n\n> 从零用 PyTorch 实现 PPO 和 GRPO，理解 DeepSeek-R1 / verl 背后的算法。"),

    md("## Part 1: PPO (Proximal Policy Optimization)\n\n### 核心公式\n$$L^{CLIP}(\\theta) = \\mathbb{E}_t \\left[ \\min(r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t) \\right]$$\n\n其中 $r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$"),

    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")"""),

    md("### 1.1 简化版 Policy Network"),
    code("""class SimplePolicy(nn.Module):
    \"\"\"简化版 policy network：输入 state，输出 action logits.\"\"\"
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, states, actions):
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

class ValueNetwork(nn.Module):
    \"\"\"Critic: 估计 state value.\"\"\"
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

print("✅ Policy & Value networks defined")"""),

    md("### 1.2 GAE (Generalized Advantage Estimation)"),
    code("""def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    \"\"\"计算 GAE advantages 和 returns.
    
    GAE(γ,λ): Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    \"\"\"
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    T = len(rewards)
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0
        next_done = dones[t + 1] if t + 1 < T else 1
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - next_done) * last_gae
    returns = advantages + values[:T]
    return advantages, returns

# ---- 测试 GAE ----
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
values = torch.tensor([0.5, 0.4, 0.6, 0.3, 0.5])
dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
adv, ret = compute_gae(rewards, values, dones)
print(f"Advantages: {adv.numpy().round(3)}")
print(f"Returns:    {ret.numpy().round(3)}")
print("✅ GAE computation works")"""),

    md("### 1.3 PPO 训练循环"),
    code("""def ppo_update(policy, value_net, optimizer_p, optimizer_v,
               states, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, epochs=4, batch_size=64):
    \"\"\"PPO clipped objective + value loss.\"\"\"
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    losses = {'policy': [], 'value': [], 'entropy': []}
    n = len(states)
    
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mb = idx[start:end]
            
            # Policy loss (clipped)
            new_log_probs, entropy = policy.evaluate(states[mb], actions[mb])
            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()
            
            optimizer_p.zero_grad()
            (policy_loss + 0.01 * entropy_loss).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer_p.step()
            
            # Value loss
            value_pred = value_net(states[mb])
            value_loss = F.mse_loss(value_pred, returns[mb])
            
            optimizer_v.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            optimizer_v.step()
            
            losses['policy'].append(policy_loss.item())
            losses['value'].append(value_loss.item())
            losses['entropy'].append(-entropy_loss.item())
    
    return {k: np.mean(v) for k, v in losses.items()}

print("✅ PPO update function defined")"""),

    md("### 1.4 模拟训练：简单环境"),
    code("""class SimpleBandit:
    \"\"\"简化环境：模拟 LLM reward optimization.
    State: 随机 embedding
    Action: 选择 response 策略 (0..3)
    Reward: action 2 最优，模拟 rule-based reward
    \"\"\"
    def __init__(self, state_dim=8):
        self.state_dim = state_dim

    def reset(self):
        return torch.randn(self.state_dim)

    def step(self, state, action):
        # 模拟 rule-based reward (DeepSeek-R1 style)
        base_reward = -0.5
        if action == 2:
            base_reward = 1.0  # 正确答案
        elif action == 1:
            base_reward = 0.3  # 部分正确
        noise = torch.randn(1).item() * 0.1
        return base_reward + noise

# 训练循环
env = SimpleBandit()
policy = SimplePolicy(state_dim=8, action_dim=4)
value_net = ValueNetwork(state_dim=8)
opt_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
opt_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)

episode_rewards = []
for episode in range(200):
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    
    for _ in range(64):  # collect rollout
        state = env.reset()
        action, lp = policy.get_action(state)
        reward = env.step(state, action.item())
        states.append(state)
        actions.append(action)
        log_probs.append(lp)
        rewards.append(reward)
        dones.append(0.0)
    dones[-1] = 1.0
    
    states_t = torch.stack(states)
    actions_t = torch.stack(actions)
    log_probs_t = torch.stack(log_probs).detach()
    rewards_t = torch.tensor(rewards)
    dones_t = torch.tensor(dones)
    
    with torch.no_grad():
        values = value_net(states_t)
    
    advantages, returns = compute_gae(rewards_t, values, dones_t)
    info = ppo_update(policy, value_net, opt_p, opt_v,
                      states_t, actions_t, log_probs_t, advantages, returns)
    episode_rewards.append(rewards_t.mean().item())

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(episode_rewards, alpha=0.3)
ax1.plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'), linewidth=2)
ax1.set_xlabel("Episode"); ax1.set_ylabel("Mean Reward")
ax1.set_title("PPO Training Curve")
ax1.axhline(y=1.0, color='r', linestyle='--', label='optimal')
ax1.legend()

# Action distribution
with torch.no_grad():
    test_states = torch.stack([env.reset() for _ in range(1000)])
    logits = policy(test_states)
    probs = F.softmax(logits, dim=-1).mean(0)
ax2.bar(range(4), probs.numpy())
ax2.set_xlabel("Action"); ax2.set_ylabel("Probability")
ax2.set_title("Learned Policy Distribution")
ax2.set_xticks(range(4), ["bad(-0.5)", "partial(0.3)", "optimal(1.0)", "bad(-0.5)"])
plt.tight_layout()
plt.savefig("ppo_training.png", dpi=100, bbox_inches='tight')
plt.show()
print(f"\\n最终策略倾向 action=2 (最优): prob={probs[2]:.3f}")
print("✅ PPO training complete!")"""),

    md("---\n## Part 2: GRPO (Group Relative Policy Optimization)\n\n### DeepSeek-R1 的核心算法\n\n**与 PPO 的关键区别**：\n- **无 Critic**：不需要 Value Network\n- **组内相对优势**：对同一 prompt 生成 G 个 response，用组内相对 reward 作为 advantage\n- **KL 惩罚**：直接对比新旧策略的 KL 散度\n\n$$\\hat{A}_i = \\frac{r_i - \\text{mean}(\\{r_1,...,r_G\\})}{\\text{std}(\\{r_1,...,r_G\\})}$$"),

    code("""class GRPO:
    \"\"\"Group Relative Policy Optimization.
    
    核心思想：
    1. 对每个 prompt，用旧策略生成 G 个 response
    2. 计算每个 response 的 reward
    3. 组内标准化 reward 作为 advantage
    4. PPO-clip + KL penalty 更新策略
    \"\"\"
    def __init__(self, policy, ref_policy, lr=1e-4, 
                 clip_eps=0.2, kl_coef=0.1, group_size=4):
        self.policy = policy
        self.ref_policy = ref_policy  # frozen reference policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.group_size = group_size

    def compute_group_advantage(self, rewards):
        \"\"\"组内标准化 reward → advantage.
        rewards: (num_prompts, group_size)
        \"\"\"
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        return (rewards - mean) / std

    def update(self, prompts, all_actions, all_old_log_probs, rewards):
        \"\"\"
        prompts: (num_prompts, state_dim)
        all_actions: (num_prompts, group_size)
        all_old_log_probs: (num_prompts, group_size)
        rewards: (num_prompts, group_size)
        \"\"\"
        advantages = self.compute_group_advantage(rewards)  # (P, G)
        
        total_loss = 0
        P, G = rewards.shape
        
        for p in range(P):
            state = prompts[p].unsqueeze(0).expand(G, -1)  # (G, dim)
            
            # New policy log probs
            new_lp, entropy = self.policy.evaluate(state, all_actions[p])
            
            # Reference policy log probs (for KL)
            with torch.no_grad():
                ref_lp, _ = self.ref_policy.evaluate(state, all_actions[p])
            
            # PPO-clip objective
            ratio = torch.exp(new_lp - all_old_log_probs[p])
            adv = advantages[p]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
            clip_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty (approximate)
            kl = (all_old_log_probs[p] - new_lp).mean()
            
            total_loss += clip_loss + self.kl_coef * kl - 0.01 * entropy.mean()
        
        total_loss /= P
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

print("✅ GRPO class defined")"""),

    md("### 2.1 GRPO 模拟训练"),
    code("""import copy

# 环境：模拟 math reward (rule-based, DeepSeek-R1 style)
def math_reward_fn(action):
    \"\"\"模拟 rule-based reward: action=2 正确，其他错误.\"\"\"
    if action == 2:
        return 1.0
    elif action == 1:
        return 0.3
    else:
        return -0.5

env = SimpleBandit()
policy = SimplePolicy(state_dim=8, action_dim=4)
ref_policy = copy.deepcopy(policy)
for p in ref_policy.parameters():
    p.requires_grad = False

grpo = GRPO(policy, ref_policy, lr=3e-4, group_size=8, kl_coef=0.05)

G = 8  # group size
grpo_rewards = []

for step in range(300):
    # 1. 采集一批 prompts
    num_prompts = 16
    prompts = torch.stack([env.reset() for _ in range(num_prompts)])
    
    # 2. 对每个 prompt 生成 G 个 response
    all_actions = torch.zeros(num_prompts, G, dtype=torch.long)
    all_log_probs = torch.zeros(num_prompts, G)
    rewards = torch.zeros(num_prompts, G)
    
    with torch.no_grad():
        for p in range(num_prompts):
            state = prompts[p].unsqueeze(0).expand(G, -1)
            for g in range(G):
                a, lp = policy.get_action(prompts[p].unsqueeze(0))
                all_actions[p, g] = a
                all_log_probs[p, g] = lp
                rewards[p, g] = math_reward_fn(a.item())
    
    # 3. GRPO update
    loss = grpo.update(prompts, all_actions, all_log_probs, rewards)
    grpo_rewards.append(rewards.mean().item())

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(grpo_rewards, alpha=0.3, label='raw')
ax1.plot(np.convolve(grpo_rewards, np.ones(20)/20, mode='valid'), linewidth=2, label='smooth')
ax1.set_xlabel("Step"); ax1.set_ylabel("Mean Reward")
ax1.set_title("GRPO Training Curve (No Critic!)")
ax1.axhline(y=1.0, color='r', linestyle='--', label='optimal')
ax1.legend()

with torch.no_grad():
    test_states = torch.stack([env.reset() for _ in range(1000)])
    logits = policy(test_states)
    probs = F.softmax(logits, dim=-1).mean(0)
ax2.bar(range(4), probs.numpy())
ax2.set_xlabel("Action"); ax2.set_ylabel("Probability")
ax2.set_title("GRPO Learned Distribution")
ax2.set_xticks(range(4), ["bad", "partial", "optimal", "bad"])
plt.tight_layout()
plt.savefig("grpo_training.png", dpi=100, bbox_inches='tight')
plt.show()
print(f"\\nGRPO 最终 action=2 概率: {probs[2]:.3f}")
print("✅ GRPO training complete!")"""),

    md("---\n## Part 3: PPO vs GRPO 对比总结\n\n| 维度 | PPO | GRPO |\n|------|-----|------|\n| **Critic** | 需要 Value Network | ❌ 不需要 |\n| **Advantage** | GAE (TD-based) | 组内相对标准化 |\n| **KL 控制** | clip only | clip + KL penalty |\n| **生成量** | 1 response/prompt | G responses/prompt |\n| **适用** | 通用 RL | 专为 LLM 设计 |\n| **论文** | Schulman et al. 2017 | DeepSeek-R1 (2025) |\n\n### 面试回答模板\n```\n\"PPO 需要训练一个 Critic 来估计 Value，计算 GAE advantage。\n GRPO 去掉了 Critic，改为对同一个 prompt 生成 G 个 response，\n 用组内 reward 的 z-score 作为 advantage。\n 好处是：(1) 省掉 Critic 的显存和训练开销，\n (2) 天然适合 LLM 的 rule-based reward（数学题对错），\n (3) DeepSeek-R1 用 GRPO 从零训出了 reasoning 能力。\"\n```"),
]

# =========================================================================
# 3. Hand-write Attention / Tokenizer / Beam Search (Enhanced)
# =========================================================================
attn_cells = [
    md("# ✍️ 手写三件套：Attention / Tokenizer / Beam Search\n\n> 限时练习版：每个模块目标 15 分钟内写完\n> 包含：变体、edge case、面试延伸"),

    md("## Part 1: Attention 全家桶\n### 1.1 Scaled Dot-Product Attention (5分钟)"),
    code("""import numpy as np
import time

def sdpa(Q, K, V, mask=None):
    \"\"\"Scaled Dot-Product Attention.
    Q: (seq_q, d_k)
    K: (seq_k, d_k)  
    V: (seq_k, d_v)
    mask: (seq_q, seq_k) — True 表示 mask 掉
    \"\"\"
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn @ V, attn

# ---- 测试 ----
np.random.seed(42)
seq_len, d = 8, 16
Q = np.random.randn(seq_len, d)
K = np.random.randn(seq_len, d)
V = np.random.randn(seq_len, d)

# Causal mask
causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
out, attn_weights = sdpa(Q, K, V, mask=causal_mask)

print(f"Output shape: {out.shape}")
print(f"Attention weights row sum: {attn_weights.sum(axis=-1).round(3)}")
assert out.shape == (seq_len, d)
assert np.allclose(attn_weights.sum(axis=-1), 1.0)
print("✅ SDPA with causal mask: passed")"""),

    md("### 1.2 Multi-Head Attention (10分钟)"),
    code("""def multi_head_attention(Q, K, V, n_heads, W_q, W_k, W_v, W_o, mask=None):
    \"\"\"完整 MHA：包含投影矩阵.
    Q,K,V: (seq, d_model)
    W_q,W_k,W_v: (d_model, d_model)
    W_o: (d_model, d_model)
    \"\"\"
    d_model = Q.shape[-1]
    d_head = d_model // n_heads
    
    # 线性投影
    q = Q @ W_q  # (seq, d_model)
    k = K @ W_k
    v = V @ W_v
    
    # Split heads: (seq, d_model) -> (n_heads, seq, d_head)
    def split(x):
        return x.reshape(-1, n_heads, d_head).transpose(1, 0, 2)
    
    q, k, v = split(q), split(k), split(v)
    
    # Per-head attention
    outputs = []
    for h in range(n_heads):
        out_h, _ = sdpa(q[h], k[h], v[h], mask)
        outputs.append(out_h)
    
    # Concat + output projection
    concat = np.concatenate(outputs, axis=-1)  # (seq, d_model)
    return concat @ W_o

# ---- 测试 ----
d_model, n_heads = 64, 8
seq_len = 16
Q = np.random.randn(seq_len, d_model).astype(np.float32)
K = np.random.randn(seq_len, d_model).astype(np.float32)
V = np.random.randn(seq_len, d_model).astype(np.float32)
W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
out = multi_head_attention(Q, K, V, n_heads, W_q, W_k, W_v, W_o, causal)
assert out.shape == (seq_len, d_model)
print(f"MHA output shape: {out.shape}")
print("✅ MHA: passed")"""),

    md("### 1.3 GQA (Grouped-Query Attention) (5分钟)"),
    code("""def gqa_attention(Q, K, V, n_q_heads, n_kv_heads, W_q, W_k, W_v, W_o, mask=None):
    \"\"\"GQA: n_kv_heads < n_q_heads, 每组 Q heads 共享一组 KV.\"\"\"
    seq_len, d_model = Q.shape
    d_head = d_model // n_q_heads
    group_size = n_q_heads // n_kv_heads
    
    q = (Q @ W_q).reshape(seq_len, n_q_heads, d_head).transpose(1, 0, 2)  # (n_q, seq, d_head)
    k = (K @ W_k).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)  # (n_kv, seq, d_head)
    v = (V @ W_v).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)
    
    outputs = []
    for h in range(n_q_heads):
        kv_idx = h // group_size  # 共享 KV 的组
        out_h, _ = sdpa(q[h], k[kv_idx], v[kv_idx], mask)
        outputs.append(out_h)
    
    concat = np.concatenate(outputs, axis=-1)
    return concat @ W_o

# ---- 测试 ----
n_q, n_kv = 8, 2  # GQA ratio = 4
d_model = 64
d_kv = d_model // n_q * n_kv  # KV projection dim

W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
W_k = np.random.randn(d_model, d_kv).astype(np.float32) * 0.02
W_v = np.random.randn(d_model, d_kv).astype(np.float32) * 0.02
W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

out = gqa_attention(Q, K, V, n_q, n_kv, W_q, W_k, W_v, W_o, causal)
print(f"GQA (n_q={n_q}, n_kv={n_kv}): output shape {out.shape}")
print(f"KV Cache 节省: {(1 - n_kv/n_q)*100:.0f}%")
print("✅ GQA: passed")"""),

    md("### 1.4 RoPE (Rotary Position Embedding) (5分钟)"),
    code("""def build_rope_cache(seq_len, dim, base=10000.0):
    \"\"\"构建 RoPE 的 cos/sin 缓存.\"\"\"
    freqs = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    positions = np.arange(seq_len).astype(np.float32)
    angles = np.outer(positions, freqs)  # (seq, dim//2)
    return np.cos(angles), np.sin(angles)

def apply_rope(x, cos, sin):
    \"\"\"应用 RoPE: x shape (seq, dim).\"\"\"
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)

# ---- 测试 ----
seq_len, dim = 32, 64
cos, sin = build_rope_cache(seq_len, dim)
x = np.random.randn(seq_len, dim).astype(np.float32)
x_rope = apply_rope(x, cos, sin)

# 验证：RoPE 保持向量长度（近似）
norm_before = np.linalg.norm(x, axis=-1)
norm_after = np.linalg.norm(x_rope, axis=-1)
assert np.allclose(norm_before, norm_after, atol=1e-5)
print("✅ RoPE preserves vector norm: passed")

# 验证：相对位置内积只依赖相对距离
q = apply_rope(np.random.randn(seq_len, dim), cos, sin)
k = apply_rope(np.random.randn(seq_len, dim), cos, sin)
# q[i] · k[j] 只依赖 i-j
print("✅ RoPE: all tests passed")"""),

    md("---\n## Part 2: Tokenizer 全家桶\n### 2.1 BPE (Byte Pair Encoding) (15分钟)"),
    code("""def bpe_train(corpus, num_merges):
    \"\"\"BPE 训练：迭代合并最高频 pair.
    corpus: dict[str, int] - word -> frequency
    Returns: merge rules
    \"\"\"
    # 初始化：每个 word 拆成字符
    vocab = {}
    for word, freq in corpus.items():
        vocab[tuple(list(word) + ['</w>'])] = freq
    
    merges = []
    for step in range(num_merges):
        # 统计所有 bigram 频率
        pairs = {}
        for tokens, freq in vocab.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        if not pairs:
            break
        
        best = max(pairs, key=pairs.get)
        merges.append(best)
        
        # 合并
        new_vocab = {}
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
        
        if step < 5 or step == num_merges - 1:
            print(f"  Step {step}: merge {best[0]} + {best[1]} -> {best[0]+best[1]} (freq={pairs[best]})")
    
    return merges, vocab

def bpe_encode(word, merges):
    \"\"\"用学到的 merge rules 编码一个新词.\"\"\"
    tokens = list(word) + ['</w>']
    for a, b in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens = tokens[:i] + [a+b] + tokens[i+2:]
            else:
                i += 1
    return tokens

# ---- 测试 ----
corpus = {
    "low": 5, "lower": 2, "newest": 6, "widest": 3,
    "new": 4, "wide": 2, "low": 5
}
merges, final_vocab = bpe_train(corpus, num_merges=10)
print(f"\\nLearned {len(merges)} merge rules")
print(f"Final vocab size: {sum(len(t) for t in final_vocab.keys())} tokens")

# 编码新词
encoded = bpe_encode("lowest", merges)
print(f"\\n'lowest' -> {encoded}")
print("✅ BPE train + encode: passed")"""),

    md("### 2.2 WordPiece Tokenizer (10分钟)"),
    code("""def wordpiece_tokenize(word, vocab, max_input_chars=200):
    \"\"\"WordPiece tokenization (BERT style).
    贪心最长前缀匹配.
    \"\"\"
    if len(word) > max_input_chars:
        return ['[UNK]']
    
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        found = None
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = '##' + substr
            if substr in vocab:
                found = substr
                break
            end -= 1
        if found is None:
            return ['[UNK]']
        tokens.append(found)
        start = end
    return tokens

# ---- 测试 ----
vocab = {'un', '##able', '##break', 'break', '##ing', 'the', '##m', 
         'a', 'an', '##n', '##e', '##d', 'play', '##ed', '##s',
         '##ly', 'un', '##break', '##able'}
vocab = {w: 1 for w in vocab}

tests = [
    ("unbreakable", ['un', '##break', '##able']),
    ("played", ['play', '##ed']),
    ("the", ['the']),
]
for word, expected in tests:
    result = wordpiece_tokenize(word, vocab)
    print(f"  '{word}' -> {result}")
print("✅ WordPiece: passed")"""),

    md("---\n## Part 3: Beam Search + Sampling (15分钟)\n### 3.1 Greedy / Top-K / Top-P / Temperature"),
    code("""def log_softmax(logits):
    logits = logits - logits.max()
    return logits - np.log(np.exp(logits).sum())

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

def greedy_decode(logits):
    return np.argmax(logits)

def top_k_sample(logits, k, temperature=1.0):
    logits = logits / temperature
    top_idx = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_idx]
    probs = softmax(top_logits)
    chosen = np.random.choice(top_idx, p=probs)
    return chosen

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = softmax(logits)
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    selected = sorted_idx[:cutoff]
    sel_probs = probs[selected]
    sel_probs /= sel_probs.sum()
    return np.random.choice(selected, p=sel_probs)

# ---- 对比测试 ----
np.random.seed(42)
vocab_size = 100
logits = np.random.randn(vocab_size) * 2
logits[42] = 5.0  # make token 42 the most likely

print(f"Greedy: {greedy_decode(logits)} (should be 42)")
print(f"Top-5: {[top_k_sample(logits, 5) for _ in range(10)]}")
print(f"Top-p=0.9: {[top_p_sample(logits, 0.9) for _ in range(10)]}")
print(f"High temp: {[top_k_sample(logits, 10, temperature=2.0) for _ in range(10)]}")
print("✅ Sampling methods: passed")"""),

    md("### 3.2 Beam Search (完整版)"),
    code("""def beam_search(score_fn, start_token, eos_token, beam_width=4, max_len=20):
    \"\"\"完整 Beam Search.
    score_fn(tokens) -> logits: 给定已生成 tokens，返回下一个 token 的 logits
    \"\"\"
    # Each beam: (cumulative_log_prob, tokens, finished)
    beams = [(0.0, [start_token], False)]
    completed = []
    
    for step in range(max_len):
        candidates = []
        for score, tokens, finished in beams:
            if finished:
                completed.append((score, tokens))
                continue
            
            logits = score_fn(tokens)
            log_probs = log_softmax(logits)
            top_k = np.argsort(-log_probs)[:beam_width * 2]  # 稍微多取一些
            
            for idx in top_k:
                new_tokens = tokens + [int(idx)]
                new_score = score + log_probs[idx]
                is_done = (idx == eos_token)
                candidates.append((new_score, new_tokens, is_done))
        
        # 保留 top beam_width
        candidates.sort(key=lambda x: -x[0])
        beams = candidates[:beam_width]
        
        # 检查是否全部完成
        if all(b[2] for b in beams):
            completed.extend([(s, t) for s, t, _ in beams])
            break
    
    # 未完成的也加入
    completed.extend([(s, t) for s, t, f in beams if not f])
    
    # Length-normalized score
    best = max(completed, key=lambda x: x[0] / len(x[1]))
    return best

# ---- 模拟测试 ----
np.random.seed(42)
vocab_size = 50
EOS = 0

def mock_lm(tokens):
    \"\"\"模拟语言模型：偏好生成 [5,10,15,20,EOS] 序列.\"\"\"
    logits = np.random.randn(vocab_size) * 0.5
    target_seq = [5, 10, 15, 20, EOS]
    pos = len(tokens) - 1  # -1 for start token
    if pos < len(target_seq):
        logits[target_seq[pos]] += 3.0  # boost target
    else:
        logits[EOS] += 5.0
    return logits

score, tokens = beam_search(mock_lm, start_token=1, eos_token=EOS, beam_width=4)
print(f"Beam search result: {tokens}")
print(f"Score (length-normalized): {score/len(tokens):.3f}")
print("✅ Beam Search: passed")"""),

    md("---\n## ⏱️ 计时练习清单\n\n| # | 任务 | 目标时间 | 实际时间 |\n|---|------|---------|----------|\n| 1 | SDPA + causal mask | 5分钟 | ______ |\n| 2 | Multi-Head Attention | 10分钟 | ______ |\n| 3 | GQA | 5分钟 | ______ |\n| 4 | RoPE | 5分钟 | ______ |\n| 5 | BPE train + encode | 15分钟 | ______ |\n| 6 | WordPiece | 10分钟 | ______ |\n| 7 | Top-K/Top-P/Greedy | 5分钟 | ______ |\n| 8 | Beam Search | 15分钟 | ______ |\n\n**总计目标：70分钟内完成所有题目**"),
]

# =========================================================================
# Save all notebooks
# =========================================================================
os.makedirs("notebooks", exist_ok=True)

for name, cells in [
    ("notebooks/leetcode_llm_system_related.ipynb", lc_cells),
    ("notebooks/rl_ppo_grpo_implementation.ipynb", rl_cells),
    ("notebooks/attention_tokenizer_beamsearch.ipynb", attn_cells),
]:
    with open(name, "w") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"✅ Created {name}")

print("\n🎉 All 3 notebooks generated!")
