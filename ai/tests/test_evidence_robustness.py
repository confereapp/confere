import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from confere_ai.grounding import resolve
from confere_ai.instruct import acceptance, load_instructions


class EvidenceRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("robustness", scripts / "prepare_evidence_robustness.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.remove(str(scripts))

    def test_generated_targets_are_valid_including_opposite_evidence_in_one_unit(self):
        rows = self.module.robustness_rows(32)
        self.assertEqual(rows, self.module.robustness_rows(32))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            load_instructions(path)
        for split in ("train", "validation"):
            self.assertEqual({r["evaluation_profile"] for r in rows if r["split"] == split}, set(self.module.PROFILES))
        for row in rows:
            if row["evaluation_profile"] == "shared_conflict":
                result = resolve(row["target"], row["document"], row["units"], 2)["result"]
                self.assertEqual(result["status"], "evidencia_insuficiente")
                self.assertEqual(len(result["evidence"]), 2)
                self.assertEqual(result["evidence"][0]["unit"], result["evidence"][1]["unit"])

    def test_replay_preserves_complete_conversations_and_excludes_test(self):
        rows = [dict(task="conversation_identity", split=split, project_id=f"{split}-{group}", turn_index=turn)
                for split in ("train", "validation", "test") for group in range(3) for turn in range(3)]
        for row in rows:
            if row["turn_index"] == 1:
                row["task"] = "conversation_clarification"
        replay = self.module.conversation_replay(rows, train_groups=2, validation_groups=1)
        self.assertEqual(len(replay), 9)
        for project in {r["project_id"] for r in replay}:
            self.assertEqual([r["turn_index"] for r in replay if r["project_id"] == project], [0, 1, 2])
        self.assertFalse(any(r["split"] == "test" for r in replay))

    def test_strong_average_cannot_hide_failure_in_conflict_profile(self):
        rows = [dict(task="grounded_atendido", exact_match=True, valid_format=True, reference_valid=True,
                     evaluation_profile="single") for _ in range(99)]
        rows.append(dict(rows[0], exact_match=False, evaluation_profile="conflict"))
        self.assertFalse(acceptance(dict(samples=rows))["passed"])
