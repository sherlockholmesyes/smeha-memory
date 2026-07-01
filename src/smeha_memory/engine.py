# SPDX-License-Identifier: Apache-2.0
"""AgentMemoryEngine — a small facade over SmehaMemory that adds the two things a flat vector store
structurally lacks:

  1. TEMPORAL VALIDITY — `remember_fact` / `observe` record (subject, predicate, object) into a
     TemporalFactStore that auto-supersedes the prior value, so `current(s, p)` returns the LATEST
     even after the fact changed (a plain similarity index returns both the stale and new mention).

  2. MULTI-HOP CONNECTIVITY — `recall_connected` grounds the query to seed nodes by similarity, then
     TRAVERSES the graph's edges up to `hops`. This reaches an answer that is CONNECTED to the query
     but not textually SIMILAR to it (e.g. "what does Alice's teammate own?" -> Alice --teammate--> Bob
     --owns--> Redis). Pure cosine similarity is mathematically similarity-bound and cannot compose
     hops; the graph can. This is the edge over a vector-DB — not flat recall (that is the vector-DB's
     turf), but composition over relations. See tests/test_engine.py for the head-to-head.

Facts are stored as graph nodes (the subject and object entities) + a typed edge (the predicate),
AND as bitemporal facts. Entities are de-duplicated by text (the text is the node id).
"""
from __future__ import annotations

from .graph import SmehaMemory
from .temporal import TemporalFactStore


class AgentMemoryEngine:
    def __init__(self, memory: SmehaMemory | None = None, *, temporal=":memory:",
                 extract_model="mistral:7b"):
        self.memory = memory or SmehaMemory()
        self.temporal = (temporal if isinstance(temporal, TemporalFactStore)
                         else TemporalFactStore(temporal))
        self.extract_model = extract_model

    # -- write ------------------------------------------------------------------------------------
    def _node(self, text) -> str:
        text = str(text)
        if text not in self.memory.nodes:
            self.memory.learn(text, node_id=text, kind="entity")
        return text

    def remember_fact(self, subject, predicate, object, *, at=None, confidence=1.0, source=None):
        """Assert an explicit (s, p, o): graph nodes + a typed edge + a bitemporal fact (auto-supersede)."""
        sid, oid = self._node(subject), self._node(object)
        self.memory.relate(sid, oid, relation=str(predicate))
        return self.temporal.add_fact(subject, predicate, object, at=at, confidence=confidence, source=source)

    def observe(self, text, *, source="observe"):
        """Extract (s, p, o) facts from raw text (via a local LLM) and remember each. Needs ollama."""
        from .extraction import extract_facts
        facts = extract_facts(text, model=self.extract_model)
        for s, p, o in facts:
            self.remember_fact(s, p, o, source=source)
        return facts

    # -- read -------------------------------------------------------------------------------------
    def current(self, subject, predicate):
        """The CURRENT (non-superseded) object for (subject, predicate), or None."""
        fact = self.temporal.current_fact(subject, predicate)
        return fact["object"] if fact else None

    def recall(self, query, *, k=8):
        """Flat similarity recall (node texts), highest score first — the vector-DB-equivalent path."""
        return [hit.node.text for hit in self.memory.recall(query, k=k)]

    def recall_connected(self, query, *, hops=2, seeds=3, k=12):
        """Ground the query to seed nodes by similarity, then TRAVERSE edges up to `hops`. Returns the
        connected node texts — reaching answers that are related-but-not-similar (the multi-hop edge)."""
        seed_ids = [hit.node.id for hit in self.memory.recall(query, k=seeds)]
        reached = set(seed_ids)
        frontier = set(seed_ids)
        adjacency = self._adjacency()
        for _ in range(max(0, hops)):
            nxt = set()
            for nid in frontier:
                for target in adjacency.get(nid, ()):
                    if target not in reached:
                        reached.add(target)
                        nxt.add(target)
            frontier = nxt
        texts = [self.memory.nodes[nid].text for nid in reached if nid in self.memory.nodes]
        return texts[:k]

    def _adjacency(self):
        adjacency: dict[str, set[str]] = {}
        for (source, target, _relation) in self.memory.edges:
            adjacency.setdefault(source, set()).add(target)
        return adjacency
