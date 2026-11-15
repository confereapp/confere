import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_context_answers import ABSENT, CONFLICT, answer, build


class ContextAnswersTests(unittest.TestCase):
    def test_oracle_distinguishes_missing_conflicting_and_explicit_negative(self):
        self.assertEqual(answer({}, "prazo"), ABSENT)
        self.assertEqual(answer({"prazo": [7, 12]}, "prazo"), CONFLICT)
        self.assertEqual(answer({"prazo": [7, 7]}, "prazo"), "7 dias.")
        self.assertEqual(answer({"testes": ["ausente"]}, "testes"), "Não.")
        self.assertEqual(answer({"testes": ["presente"]}, "testes"), "Sim.")

    def test_targets_and_histories_follow_scenarios_without_label_in_prompt(self):
        rows = build(30, 6)
        self.assertEqual(rows, build(30, 6))
        for row in rows:
            if row["task"].endswith("memory"):
                if row["turn_index"] == 2:
                    corrected = row["history"][-1][1].removeprefix("Prazer, ")
                    self.assertEqual(row["target"], "Seu nome é " + corrected)
            else:
                self.assertEqual(row["target"], answer(row["scenario"]["facts"], row["scenario"]["field"]))
                self.assertNotIn(row["task"].removeprefix("conversation_context_answer_"), row["prompt"].splitlines()[0])
        train_prompts = {r["prompt"] for r in rows if r["split"] == "train"}
        validation_prompts = {r["prompt"] for r in rows if r["split"] == "validation"}
        self.assertFalse(train_prompts & validation_prompts)
