import tempfile
import unittest
from pathlib import Path

from smeha_memory import SmehaMemory
from smeha_memory.cli import main


class SmehaMemoryTests(unittest.TestCase):
    def test_learn_and_recall(self) -> None:
        memory = SmehaMemory()
        first = memory.learn("assets load before visible entities", kind="note", tags=("assets",))
        memory.learn("combat input should be server checked", kind="note", tags=("combat",))

        hits = memory.recall("assets visible entities", k=1)

        self.assertEqual(hits[0].node.id, first)

    def test_tags_filter_recall(self) -> None:
        memory = SmehaMemory()
        memory.learn("physics body seam handoff", kind="note", tags=("physics",))
        memory.learn("asset package manifest", kind="note", tags=("assets",))

        hits = memory.recall("manifest", tags=("assets",), k=4)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].node.tags, ("assets",))

    def test_relation_boost_uses_seed(self) -> None:
        memory = SmehaMemory()
        seed = memory.learn("dependency root", node_id="seed")
        target = memory.learn("hidden asset package", node_id="target")
        memory.learn("unrelated physics note", node_id="other")
        memory.relate(seed, target, relation="supports", weight=4.0)

        hits = memory.recall("unrelated", k=3, seed_ids=[seed], seed_weight=1.0)

        self.assertEqual(hits[0].node.id, target)
        self.assertGreater(hits[0].edge_boost, 0.0)

    def test_jsonl_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.jsonl"
            memory = SmehaMemory.open(db)
            a = memory.learn("first", node_id="a", now=1.0)
            b = memory.learn("second", node_id="b", now=2.0)
            memory.relate(a, b, relation="next", weight=0.7)
            memory.reinforce(b, amount=0.3, now=3.0)

            restored = SmehaMemory.open(db)

            self.assertEqual(set(restored.nodes), {"a", "b"})
            self.assertEqual(restored.nodes["b"].weight, 1.3)
            self.assertEqual(restored.edges[("a", "b", "next")].weight, 0.7)

    def test_status_counts(self) -> None:
        memory = SmehaMemory()
        memory.learn("a", kind="note", tags=("one",))
        memory.learn("b", kind="fact", tags=("one", "two"))

        status = memory.status()

        self.assertEqual(status["nodes"], 2)
        self.assertEqual(status["kinds"], {"fact": 1, "note": 1})
        self.assertEqual(status["tags"]["one"], 2)

    def test_cli_learn_recall_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "memory.jsonl")
            self.assertEqual(main(["--db", db, "learn", "--id", "n1", "--tag", "assets", "asset manifest"]), 0)
            self.assertEqual(main(["--db", db, "recall", "manifest", "--k", "1"]), 0)
            self.assertEqual(main(["--db", db, "status"]), 0)


if __name__ == "__main__":
    unittest.main()
