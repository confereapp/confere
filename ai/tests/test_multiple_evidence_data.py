import importlib.util
import tempfile
import json
import sys
import unittest
from pathlib import Path

from confere_ai.instruct import load_instructions


class MultipleEvidenceDataTests(unittest.TestCase):
    def test_all_modes_have_valid_targets_and_projects_stay_in_one_split(self):
        scripts = Path(__file__).parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("multiple_evidence", scripts / "prepare_multiple_evidence.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rows = module.documents(24)
        finally:
            sys.path.remove(str(scripts))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            loaded, _ = load_instructions(path)
        self.assertEqual(len(loaded), 48)
        for split in ("train", "validation"):
            self.assertEqual({r["mode"] for r in loaded if r["split"] == split}, set(range(6)))
        self.assertFalse(any(r["split"] == "test" for r in loaded))
