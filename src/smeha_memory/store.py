"""Append-only JSONL event store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping


class AppendOnlyStore:
    """A small append-only JSONL floor."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")

    def events(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

