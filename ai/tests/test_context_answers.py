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


class ContextAuditTests(unittest.TestCase):
    def test_family_regression_is_not_hidden_by_new_skill_gain(self):
        from audit_context_answers import audit
        samples = [dict(id=str(i), prompt=str(i), expected="ok", task="conversation_context_answer_"+task, exact_match=True)
                   for i, task in enumerate(["fact", "missing", "conflict", "support", "memory"])]
        samples += [dict(id="legacy", prompt="old", expected="ok", task="conversation_identity", exact_match=True),
                    dict(id="doc", prompt="doc", expected="ok", task="grounded_atendido", exact_match=True,
                         valid_format=True, reference_valid=True)]
        after = [dict(row) for row in samples]
        after[-2]["exact_match"] = False
        result = audit(dict(baseline=dict(samples=samples), selected=dict(samples=after), test_examples_used=0))
        self.assertTrue(result["context_pass"])
        self.assertFalse(result["controlled_criteria_pass"])
        self.assertIn("conversation_identity", result["regressions"])


class ExpandedContextTests(unittest.TestCase):
    def test_names_are_compositional_and_validation_is_only_explicitly_corrected(self):
        from expand_context_answers import expand, HELD_NAMES
        original = build(30, 6)
        expanded, corrections = expand(original)
        old_validation = {r["id"]: r for r in original if r["split"] == "validation"}
        for row in expanded:
            if row["split"] == "validation":
                old = old_validation[row["scenario"].get("source_id", row["id"])]
                self.assertEqual(row["target"], old["target"])
                self.assertEqual(row["history"], old["history"])
                if not row["id"].startswith("distractor-"):
                    self.assertEqual(row["prompt"] == old["prompt"], row["id"] not in corrections)
            elif row["task"].endswith("memory"):
                self.assertFalse(set(row["scenario"]["names"]) & HELD_NAMES)
                if row["scenario"]["question_role"] == "user":
                    self.assertEqual(row["target"], "Seu nome é " + row["scenario"]["names"][1] + ".")
                if row["scenario"]["question_role"] == "assistant": self.assertEqual(row["target"], "Sou o Confere.")
        questions = {r["prompt"].splitlines()[-1] for r in expanded if r["split"] == "validation"}
        self.assertFalse(any(r["prompt"].splitlines()[-1] in questions for r in expanded if r["split"] == "train"))
