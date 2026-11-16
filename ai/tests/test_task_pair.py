import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_task_pair import combine


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
