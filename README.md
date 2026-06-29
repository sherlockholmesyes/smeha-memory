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

## Install

From a checkout:

```bash
pip install -e .
```

Or use it directly by setting `PYTHONPATH=src`.

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

## Scope

This package is a small reference component. It is not a vector database, not a
large-model runtime, and not a substitute for external verification. Use it as
a readable base for local experiments and adapters.

