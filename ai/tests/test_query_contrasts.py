import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_query_contrasts import conversations, documents
from prepare_context_answers import answer
from prepare_lexical_binding import expected


class QueryContrastTests(unittest.TestCase):
    def test_same_document_requires_different_answers_without_lexical_shortcuts(self):
        rows = documents(30)
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual(a["units"], b["units"])
            for row in [a, b]:
                self.assertEqual(json.loads(row["target"]), expected(row["scenario"]["units"], row["scenario"]["feature"]))
        self.assertTrue(any(json.loads(a["target"]) != json.loads(b["target"]) for a, b in zip(rows[::2], rows[1::2])))

    def test_unrelated_conflict_does_not_contaminate_other_questions(self):
        rows = conversations(24)
        for row in rows:
            self.assertEqual(row["target"], answer(row["scenario"]["facts"], row["scenario"]["field"]))
        for offset in range(0, len(rows), 4):
            group = rows[offset:offset+4]
            self.assertEqual(len({r["prompt"].rsplit("\n", 1)[0] for r in group}), 1)
            self.assertEqual(len({r["scenario"]["field"] for r in group}), 4)
        conflict_groups = [rows[i:i+4] for i in range(0, len(rows), 4) if any(r["task"].endswith("conflict") for r in rows[i:i+4])]
        self.assertTrue(conflict_groups)
        self.assertTrue(all(sum(r["task"].endswith("conflict") for r in g) == 1 for g in conflict_groups))
