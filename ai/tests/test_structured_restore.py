import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_structured_answers import restore
from prepare_context_answers import build


class StructuredRestoreTests(unittest.TestCase):
    def test_only_historical_training_facts_are_restored(self):
        history = build(3, 2)
        reserved = [r for r in history if r["split"] == "validation"]
        rows, added = restore(reserved, [history])
        self.assertEqual(added, 9)
        self.assertEqual([r for r in rows if r["split"] == "validation"], reserved)
        self.assertTrue(all(r["split"] == "train" and r["id"].startswith("restored-") for r in rows[len(reserved):]))
