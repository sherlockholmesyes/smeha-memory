"""Deterministic local embedding helpers.

The hashing embedder is intentionally simple. It is useful for examples,
tests, and offline adapters. Production systems can supply their own embedder
with the same `embed(text) -> tuple[float, ...]` shape.
"""

from __future__ import annotations

import hashlib
import re
from math import sqrt
from typing import Protocol


Vector = tuple[float, ...]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    def embed(self, text: str) -> Vector:
        """Return a vector for text."""


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def l2_normalize(values: list[float]) -> Vector:
    length = sqrt(sum(value * value for value in values))
    if length == 0.0:
        return tuple(values)
    return tuple(value / length for value in values)


def cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right))


class HashingEmbedder:
    """A deterministic bag-of-tokens embedder."""

    def __init__(self, dimensions: int = 256, salt: str = "smeha-memory-v1") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.salt = salt

    def embed(self, text: str) -> Vector:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(
                f"{self.salt}:{token}".encode("utf-8"),
                digest_size=16,
            ).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign
        return l2_normalize(values)

