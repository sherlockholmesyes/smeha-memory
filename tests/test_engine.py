# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentMemoryEngine: temporal supersede + the multi-hop connectivity edge.

Both are deterministic (the default HashingEmbedder, no network / no LLM)."""

from smeha_memory.engine import AgentMemoryEngine


def test_temporal_supersede_returns_latest():
    eng = AgentMemoryEngine()
    eng.remember_fact("project", "uses_storage", "Redis")
    assert eng.current("project", "uses_storage") == "Redis"
    eng.remember_fact("project", "uses_storage", "SQLite")          # the fact changed
    assert eng.current("project", "uses_storage") == "SQLite"       # returns the LATEST, not both
    # exactly one open fact remains for (s, p) after a correct supersede
    assert len(eng.temporal.open_facts("project", "uses_storage")) == 1


def test_multihop_reaches_what_flat_recall_misses():
    """'What does Alice's teammate own?' — the answer is 2 hops away and shares no words with the
    query. Flat similarity recall misses it; multi-hop traversal reaches it."""
    eng = AgentMemoryEngine()
    eng.remember_fact("Alice", "teammate", "Bob")
    eng.remember_fact("Bob", "owns", "Redis cluster")               # the 2-hop answer
    # distractors (more query-similar, but wrong)
    eng.remember_fact("Alice", "owns", "Titan dashboard")          # Alice's OWN (right person, wrong relation)
    eng.remember_fact("Carol", "teammate", "Dave")
    eng.remember_fact("Dave", "owns", "MySQL database")
    eng.remember_fact("Erin", "teammate", "Frank")
    eng.remember_fact("Frank", "owns", "Mongo shard")

    query = "Alice teammate"
    flat = eng.recall(query, k=3)                                   # the vector-DB-equivalent path
    connected = eng.recall_connected(query, hops=2, seeds=3)        # the graph-traversal path

    assert "Redis cluster" not in flat                             # cosine can't compose hops
    assert "Redis cluster" in connected                            # traversal reaches it


if __name__ == "__main__":
    test_temporal_supersede_returns_latest()
    test_multihop_reaches_what_flat_recall_misses()
    print("PASS")
