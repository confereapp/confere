import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_open_binding import build
from prepare_lexical_binding import documents, expected


class OpenBindingTests(unittest.TestCase):
    def test_pairs_keep_identical_sources_and_remove_only_ambiguous_training_rows(self):
        source = documents(1)
        reserved = copy.deepcopy(source[0]); reserved.update(id="reserved", project_id="reserved", split="validation")
        ambiguous = copy.deepcopy(source[0]); ambiguous.update(id="ambiguous", project_id="ambiguous")
        ambiguous["prompt"] += " A especificação prevê testes."
        source += [reserved, ambiguous]
        rows, removed = build(source, 18)
        self.assertEqual(removed, ["ambiguous"])
        self.assertEqual([r for r in rows if r["split"] == "validation"], [reserved])
        added = [r for r in rows if r["id"].startswith("open-")]
        self.assertEqual(len(added), 36)
        for first, second in zip(added[::2], added[1::2]):
            self.assertEqual(first["units"], second["units"])
            self.assertNotEqual(first["requirement"], second["requirement"])
            for row in [first, second]:
                self.assertEqual(json.loads(row["target"]), expected(row["scenario"]["units"], row["scenario"]["feature"]))
