"""Command-line interface for SMEHA Memory."""

from __future__ import annotations

import argparse
import json

from .graph import SmehaMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smeha-memory")
    parser.add_argument("--db", default="smeha_memory.jsonl", help="JSONL memory path")
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn", help="Add a node")
    learn.add_argument("text")
    learn.add_argument("--kind", default="note")
    learn.add_argument("--tag", action="append", default=[])
    learn.add_argument("--id", dest="node_id")
    learn.add_argument("--related", action="append", default=[])

    recall = sub.add_parser("recall", help="Recall nodes")
    recall.add_argument("query")
    recall.add_argument("--k", type=int, default=8)
    recall.add_argument("--kind")
    recall.add_argument("--tag", action="append", default=[])
    recall.add_argument("--seed", action="append", default=[])

    relate = sub.add_parser("relate", help="Add a weighted edge")
    relate.add_argument("source")
    relate.add_argument("target")
    relate.add_argument("--relation", default="related")
    relate.add_argument("--weight", type=float, default=1.0)

    reinforce = sub.add_parser("reinforce", help="Adjust node weight")
    reinforce.add_argument("node_id")
    reinforce.add_argument("--amount", type=float, default=0.1)

    sub.add_parser("status", help="Print memory status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    memory = SmehaMemory.open(args.db)

    if args.command == "learn":
        node_id = memory.learn(
            args.text,
            kind=args.kind,
            tags=args.tag,
            node_id=args.node_id,
            related_ids=args.related,
        )
        print(node_id)
        return 0

    if args.command == "recall":
        hits = memory.recall(
            args.query,
            k=args.k,
            kind=args.kind,
            tags=args.tag,
            seed_ids=args.seed,
        )
        print(
            json.dumps(
                [
                    {
                        "id": hit.node.id,
                        "kind": hit.node.kind,
                        "tags": list(hit.node.tags),
                        "score": hit.score,
                        "similarity": hit.similarity,
                        "edge_boost": hit.edge_boost,
                        "text": hit.node.text,
                    }
                    for hit in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "relate":
        memory.relate(args.source, args.target, relation=args.relation, weight=args.weight)
        return 0

    if args.command == "reinforce":
        memory.reinforce(args.node_id, amount=args.amount)
        return 0

    if args.command == "status":
        print(json.dumps(memory.status(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")

