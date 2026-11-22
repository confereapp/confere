import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from confere_ai.instruct import load_instructions


class EvidenceIntensiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("intensive", scripts / "prepare_evidence_intensive.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.remove(str(scripts))

    def test_counterfactual_targets_are_valid_and_only_training_is_generated(self):
        rows = self.module.intensive_rows(40)
        self.assertEqual(rows, self.module.intensive_rows(40))
        self.assertEqual({r["split"] for r in rows}, {"train"})
        self.assertEqual({r["evaluation_profile"] for r in rows}, set(self.module.PROFILES))
        self.assertEqual(len({r["prompt"].casefold() for r in rows}), len(rows))
        # O carregador exige validação; uma família inteira é movida apenas neste teste de contrato.
        for row in rows:
            if row["project_id"] == "intensive-0":
                row["split"] = "validation"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            load_instructions(path)
        for row in rows:
            if row["evaluation_profile"] == "shared_conflict":
                target = json.loads(row["target"])
                self.assertEqual(target["status"], "evidencia_insuficiente")
                self.assertEqual(len(target["evidence"]), 2)
                self.assertEqual(target["evidence"][0]["unit"], target["evidence"][1]["unit"])
