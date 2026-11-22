import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_natural_answers import build, answer, HELD_NAMES, ABSENT, CONFLICT
from evaluate_context_smoke import CASES


class NaturalAnswersTests(unittest.TestCase):
    def test_targets_follow_facts_and_roles_without_copying_diagnostics(self):
        rows = build([], 60, 20, 12)
        self.assertEqual(rows, build([], 60, 20, 12))
        self.assertEqual(len(rows), 92)
        held_prompts = {case["message"] for case in CASES}
        for row in rows:
            self.assertEqual(row["split"], "train")
            self.assertNotIn(row["prompt"], held_prompts)
            scenario = row["scenario"]
            if "facts" in scenario:
                self.assertEqual(row["target"], answer(scenario["facts"], scenario["field"]))
                self.assertFalse(HELD_NAMES & {v for k in ["nome", "responsavel"] for v in scenario["facts"].get(k, [])})
            elif "assignments" in scenario:
                self.assertEqual(row["target"], scenario["assignments"][scenario["requested"]] + ".")
                self.assertFalse(HELD_NAMES & set(scenario["assignments"].values()))
            else:
                self.assertEqual(row["target"], ["Sim.", "Não.", ABSENT, CONFLICT][scenario["state"]])
