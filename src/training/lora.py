"""A tiny LoRA linear layer implementation in NumPy."""

from __future__ import annotations

import numpy as np


class LoRALinear:
    """Frozen base weight + trainable low-rank update.

    delta_W = (alpha / r) * (A @ B), where:
    - A: [in_features, r]
    - B: [r, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        seed: int = 0,
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        rng = np.random.default_rng(seed)
        # Base weight is treated as frozen in LoRA fine-tuning.
        self.weight = rng.normal(size=(in_features, out_features)).astype(np.float32) * 0.02
        # LoRA init: A random, B zero -> starts from base model behavior.
        self.lora_a = rng.normal(size=(in_features, rank)).astype(np.float32) * 0.01
        self.lora_b = np.zeros((rank, out_features), dtype=np.float32)

    def delta_weight(self) -> np.ndarray:
        return self.scaling * (self.lora_a @ self.lora_b)

    def merged_weight(self) -> np.ndarray:
        return self.weight + self.delta_weight()

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.in_features:
            raise ValueError("x last dim must equal in_features")
        return x @ self.merged_weight()


if __name__ == "__main__":
    layer = LoRALinear(in_features=16, out_features=32, rank=4, alpha=8)
    x = np.random.randn(2, 10, 16).astype(np.float32)
    y = layer.forward(x)
    print("output shape:", y.shape)
