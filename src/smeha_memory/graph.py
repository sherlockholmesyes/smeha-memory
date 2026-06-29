"""Typed co-activation memory graph."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .embeddings import Embedder, HashingEmbedder, Vector, cosine
from .store import AppendOnlyStore


@dataclass(frozen=True)
class MemoryNode:
    id: str
    text: str
    vector: Vector
    kind: str
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]
    weight: float
    created_at: float
    updated_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "vector": list(self.vector),
            "kind": self.kind,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "weight": self.weight,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MemoryNode":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            vector=tuple(float(value) for value in data["vector"]),
            kind=str(data.get("kind") or "note"),
            tags=tuple(str(tag) for tag in data.get("tags") or ()),
            metadata=dict(data.get("metadata") or {}),
            weight=float(data.get("weight", 1.0)),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass(frozen=True)
class MemoryEdge:
    source: str
    target: str
    relation: str
    weight: float

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MemoryEdge":
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            relation=str(data.get("relation") or "related"),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class RecallHit:
    node: MemoryNode
    similarity: float
    edge_boost: float
    score: float


class SmehaMemory:
    """In-memory graph with optional append-only persistence."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        store: AppendOnlyStore | None = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = store
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[tuple[str, str, str], MemoryEdge] = {}

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
    ) -> "SmehaMemory":
        store = AppendOnlyStore(path)
        memory = cls(embedder=embedder, store=store)
        memory._replay(store)
        return memory

    def learn(
        self,
        text: str,
        *,
        kind: str = "note",
        tags: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        node_id: str | None = None,
        weight: float = 1.0,
        related_ids: Iterable[str] = (),
        relation: str = "related",
        now: float | None = None,
    ) -> str:
        node_id = node_id or uuid4().hex
        if node_id in self.nodes:
            raise ValueError(f"node already exists: {node_id}")
        now = time.time() if now is None else float(now)
        node = MemoryNode(
            id=node_id,
            text=str(text),
            vector=self.embedder.embed(text),
            kind=str(kind),
            tags=tuple(dict.fromkeys(str(tag) for tag in tags)),
            metadata=dict(metadata or {}),
            weight=float(weight),
            created_at=now,
            updated_at=now,
        )
        self.nodes[node_id] = node
        self._append({"op": "learn", "node": node.to_json()})
        for related_id in related_ids:
            self.relate(node_id, str(related_id), relation=relation, weight=1.0)
            self.relate(str(related_id), node_id, relation=relation, weight=1.0)
        return node_id

    def relate(
        self,
        source: str,
        target: str,
        *,
        relation: str = "related",
        weight: float = 1.0,
    ) -> None:
        self._require(source)
        self._require(target)
        key = (str(source), str(target), str(relation))
        old = self.edges.get(key)
        edge = MemoryEdge(
            source=str(source),
            target=str(target),
            relation=str(relation),
            weight=(old.weight if old else 0.0) + float(weight),
        )
        self.edges[key] = edge
        self._append({"op": "relate", "edge": edge.to_json()})

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        kind: str | None = None,
        tags: Iterable[str] = (),
        seed_ids: Iterable[str] = (),
        seed_weight: float = 0.15,
        min_score: float | None = None,
    ) -> list[RecallHit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        query_vec = self.embedder.embed(query)
        required_tags = set(str(tag) for tag in tags)
        seeds = tuple(str(seed_id) for seed_id in seed_ids)
        hits: list[RecallHit] = []
        for node in self.nodes.values():
            if kind is not None and node.kind != kind:
                continue
            if required_tags and not required_tags.issubset(set(node.tags)):
                continue
            similarity = cosine(query_vec, node.vector)
            edge_boost = self._edge_boost(seeds, node.id) * seed_weight
            score = similarity * node.weight + edge_boost
            if min_score is None or score >= min_score:
                hits.append(RecallHit(node=node, similarity=similarity, edge_boost=edge_boost, score=score))
        hits.sort(key=lambda hit: (hit.score, hit.node.updated_at, hit.node.id), reverse=True)
        return hits[:k]

    def reinforce(self, node_id: str, *, amount: float = 0.1, now: float | None = None) -> None:
        node = self._require(node_id)
        now = time.time() if now is None else float(now)
        updated = replace(node, weight=max(0.0, node.weight + float(amount)), updated_at=now)
        self.nodes[node.id] = updated
        self._append({"op": "reinforce", "id": node.id, "amount": float(amount), "now": now})

    def decay(self, *, rate: float = 0.01, now: float | None = None) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        now = time.time() if now is None else float(now)
        for node in list(self.nodes.values()):
            self.nodes[node.id] = replace(
                node,
                weight=max(0.0, node.weight * (1.0 - rate)),
                updated_at=now,
            )
        for key, edge in list(self.edges.items()):
            new_weight = edge.weight * (1.0 - rate)
            if abs(new_weight) < 1e-12:
                del self.edges[key]
            else:
                self.edges[key] = replace(edge, weight=new_weight)
        self._append({"op": "decay", "rate": float(rate), "now": now})

    def status(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        tags: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
            for tag in node.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "kinds": dict(sorted(kinds.items())),
            "tags": dict(sorted(tags.items())),
        }

    def _edge_boost(self, seeds: Iterable[str], target: str) -> float:
        total = 0.0
        for seed in seeds:
            for (source, edge_target, _relation), edge in self.edges.items():
                if source == seed and edge_target == target:
                    total += edge.weight
        return total

    def _append(self, event: Mapping[str, Any]) -> None:
        if self.store is not None:
            self.store.append(event)

    def _replay(self, store: AppendOnlyStore) -> None:
        original_store = self.store
        self.store = None
        try:
            for event in store.events():
                self._apply_event(event)
        finally:
            self.store = original_store

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        op = event.get("op")
        if op == "learn":
            node = MemoryNode.from_json(event["node"])
            self.nodes[node.id] = node
        elif op == "relate":
            edge = MemoryEdge.from_json(event["edge"])
            key = (edge.source, edge.target, edge.relation)
            self.edges[key] = edge
        elif op == "reinforce":
            self.reinforce(str(event["id"]), amount=float(event["amount"]), now=float(event["now"]))
        elif op == "decay":
            self.decay(rate=float(event["rate"]), now=float(event["now"]))
        else:
            raise ValueError(f"unknown event op: {op!r}")

    def _require(self, node_id: str) -> MemoryNode:
        try:
            return self.nodes[str(node_id)]
        except KeyError as exc:
            raise KeyError(f"unknown node: {node_id}") from exc

