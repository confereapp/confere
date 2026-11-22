import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_task_pair import combine, original_partition


class TaskPairTests(unittest.TestCase):
    def test_joint_result_keeps_every_original_case_and_rejects_changed_targets(self):
        samples = [dict(id=str(i), task=task, prompt=str(i), expected="ok", exact_match=True)
                   for i, task in enumerate(["grounded_atendido", "conversation_definition"])]
        baseline = dict(samples=samples)
        parts = [dict(total=1, correct=1, samples=[row]) for row in samples]
        self.assertEqual(combine(baseline, parts[::-1])["samples"], samples)
        for invalid in [parts[:1], parts + parts[:1]]:
            with self.assertRaises(ValueError): combine(baseline, invalid)
        changed = copy.deepcopy(parts); changed[0]["samples"][0]["expected"] = "outro alvo"
        with self.assertRaises(ValueError): combine(baseline, changed)

    def test_extra_development_cases_cannot_replace_or_modify_original_partition(self):
        original = [dict(id="old", task="grounded_atendido", prompt="old", target="yes", split="validation")]
        extra = dict(id="development-new", task="grounded_atendido", prompt="new", target="yes", split="validation")
        self.assertEqual(original_partition(original, original+[extra]), [extra])
        for subset in [[extra], [dict(original[0], target="no"), extra], original+[dict(extra, split="test")], original+[dict(extra, id="unmarked")]]:
            with self.assertRaises(ValueError): original_partition(original, subset)

    def test_natural_development_conversations_do_not_replace_original_cases(self):
        original = [dict(id="old", task="conversation_identity", split="validation")]
        extra = dict(id="development-natural-0", task="conversation_context_answer_natural", split="validation")
        self.assertEqual(original_partition(original, original+[extra]), [extra])
        with self.assertRaises(ValueError): original_partition(original, [extra])
