import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_query_contrasts import conversations, documents, long_documents, intent_variations
from confere_ai.tokenizer import ByteBPE
from confere_ai.grounding import judge
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

    def test_long_transfer_uses_actual_fragments_and_never_labels_partial_sentences(self):
        rows, skipped = long_documents(ByteBPE([]), 1)
        self.assertTrue(rows)
        self.assertGreaterEqual(skipped, 0)
        self.assertTrue(all(r["split"] == "train" and r["id"].startswith("query-long-") for r in rows))
        self.assertTrue(all(judge(r["target"], r)["exact_match"] for r in rows))
        self.assertTrue(any(r["training_profile"] == "scope_long" for r in rows))

    def test_intent_variations_keep_target_and_do_not_copy_validation_requests(self):
        source = [dict(id="train", project_id="train", split="train", task="conversation_slang_function",
                       prompt="o que você faz?", target="Sou um experimento para apoiar a conferência de projetos.", history=[], turn_index=0),
                  dict(id="val", project_id="val", split="validation", task="conversation_slang_function",
                       prompt="o q tu faz aí?", target="Sou um experimento para apoiar a conferência de projetos.", history=[], turn_index=0)]
        rows = intent_variations(source, 40)
        self.assertTrue(rows)
        self.assertTrue(all(r["prompt"] != source[1]["prompt"] for r in rows))
        self.assertTrue(all(r["split"] == "train" for r in rows))
        transformed = [r for r in rows if r["id"].startswith("intent-variation-")]
        self.assertTrue(transformed)
        self.assertTrue(all(r["target"] == source[0]["target"] and r["turn_index"] == 0 for r in transformed))
