# SMEHA Memory

Clean Apache-2.0 reference implementation of a typed co-activation memory graph
for local software-agent experiments.

This is a sanitized public package. It contains no private database, no logs,
no credentials, no internal notes, and no project-specific memory dump.

## What It Provides

- `SmehaMemory`: learn, recall, relate, reinforce, decay, and inspect status.
- Typed nodes with tags and metadata.
- Weighted directed relation edges.
- Deterministic local hashing embedder, so examples work without an API key.
- Append-only JSONL floor and deterministic replay.
- CLI for quick local use.

## How It Works

SMEHA Memory keeps two layers:

- an in-memory graph of nodes and weighted typed edges for fast recall
- an append-only JSONL event log that can replay the graph exactly

The default embedder is a deterministic hashing embedder. That keeps the package
offline, reproducible, and key-free. If you need stronger semantic recall, pass
your own embedder object with an `embed(text) -> tuple[float, ...]` method.

## Install

From a checkout:

```bash
pip install -e .
```

Or use it directly by setting `PYTHONPATH=src`.

## Quick Start

```bash
pip install -e .
python -m smeha_memory learn --db local_memory.jsonl "Durable facts belong in the append-only floor."
python -m smeha_memory recall --db local_memory.jsonl "durable facts"
python -m smeha_memory status --db local_memory.jsonl
```

The `local_memory.jsonl` file is your private memory log. Do not commit it
unless you intentionally want to publish its contents.

## Python Usage

```python
from smeha_memory import SmehaMemory

memory = SmehaMemory.open("local_memory.jsonl")

alpha = memory.learn(
    "The renderer should only publish frames after assets are available.",
    kind="engineering_note",
    tags=("rendering", "assets"),
)

beta = memory.learn(
    "Missing dependencies should produce placeholders or block visibility.",
    kind="engineering_note",
    tags=("assets", "dependency"),
)

memory.relate(alpha, beta, relation="supports", weight=0.8)

for hit in memory.recall("asset visibility dependency", k=3):
    print(round(hit.score, 3), hit.node.text)
```

## Engine: Temporal Validity And Multi-Hop Connectivity

`AgentMemoryEngine` wraps `SmehaMemory` and adds the two things a flat vector store
structurally cannot do.

**Temporal validity.** Facts change. `current(subject, predicate)` returns the LATEST
value; a plain similarity index returns both the stale and the new mention and leaves
the model to guess which is current.

```python
from smeha_memory import AgentMemoryEngine

eng = AgentMemoryEngine()
eng.remember_fact("project", "uses_storage", "Redis")
eng.remember_fact("project", "uses_storage", "SQLite")   # the fact changed
eng.current("project", "uses_storage")                   # -> "SQLite" (latest, not both)
```

**Multi-hop connectivity.** Some answers are connected to the query but not textually
similar to it. Cosine similarity is similarity-bound and cannot compose hops; a graph
can. `recall_connected` grounds the query to seed nodes, then traverses edges.

```python
eng.remember_fact("Alice", "teammate", "Bob")
eng.remember_fact("Bob", "owns", "Redis cluster")        # the answer, two hops away

eng.recall("Alice teammate", k=3)                        # flat: does NOT surface "Redis cluster"
eng.recall_connected("Alice teammate", hops=2)           # traversal: reaches "Redis cluster"
```

This is where a graph memory earns its keep over a vector database: not flat recall
(that is the vector database's strength) but composition over relations. See
`tests/test_engine.py` for the head-to-head.

**Extraction (optional).** `observe(text)` extracts `(subject, predicate, object)` facts
from raw text via a local ollama model and records each with temporal validity. It needs
a running ollama; everything else works offline with the default hashing embedder.

## Agent Usage Pattern

Use it as a small durable memory layer around an agent:

1. Call `recall(query)` before a non-trivial step to retrieve local context.
2. Do the work against the real source of truth: files, tests, logs, or tools.
3. Call `learn(text, tags=...)` after a material result, decision, or failure.
4. Use `relate(source, target, relation=...)` when two records should reinforce
   each other on future recalls.

Keep private memory files out of git. The library stores user-provided text in
plain JSONL by design so that the log is inspectable and easy to back up.

## CLI Usage

Learn:

```bash
python -m smeha_memory learn --db local_memory.jsonl --kind note --tag assets "Assets must load before visibility."
```

Recall:

```bash
python -m smeha_memory recall --db local_memory.jsonl "asset visibility" --k 5
```

Status:

```bash
python -m smeha_memory status --db local_memory.jsonl
```

Relate two records:

```bash
python -m smeha_memory relate --db local_memory.jsonl SOURCE_ID TARGET_ID --relation supports --weight 0.8
```

## MCP / Agent Integration

This repository intentionally ships no private MCP configuration. To integrate
it with an agent runtime, wrap `SmehaMemory.open(path)` behind your local
tooling and expose three operations:

- `learn(text, kind=None, tags=[])`
- `recall(query, k=8, tags=[])`
- `status()`

Keep the JSONL file local unless you explicitly choose to share it.

## Privacy And Safety

- The package ships code only: no private database, exported graph, chat log, or
  runtime configuration.
- The default embedder is local and deterministic, so examples do not call any
  external API.
- Memory text is stored in plain JSONL for auditability. Put the JSONL file in a
  private path, back it up deliberately, and keep it out of git by default.
- If you add an MCP wrapper or stronger embedder, keep credentials in your own
  secret store, not in this repository.

## Tests

From a checkout without installing:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"; python -m unittest discover -s tests -v
```

## Scope

This package is a small reference component. It is not a vector database, not a
large-model runtime, and not a substitute for external verification. Use it as
a readable base for local experiments and adapters.
