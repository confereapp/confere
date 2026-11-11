import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path

from confere_ai.project_review import review_extracted
from confere_ai.tokenizer import ByteBPE


class ApiBatchDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("api_batches", scripts / "prepare_api_batches.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.remove(str(scripts))

    def audit_oracle(self, scene, rows):
        targets = iter(r["target"] for r in rows)
        result = review_extracted([scene["document"]], scene["requirements"], ByteBPE([]), 1024,
                                  lambda ids: next(targets), schema_version=2, response_tokens=256)
        self.assertEqual([(r["status"], r["conflicting_evidence"]) for r in result["requirements"]],
                         [(r["status"], r["conflicting_evidence"]) for r in scene["expected"]])
        text = scene["document"]["units"][0]["text"]
        for row in result["requirements"]:
            self.assertTrue(row["model_outputs_valid"])
            self.assertEqual(row["coverage"]["processed_characters"], len(text))
            for evidence in row["evidence"]:
                self.assertEqual(evidence["quote"], text[evidence["char_start"]:evidence["char_end"]])
        self.assertIsNone(next(targets, None))

    def test_same_source_has_opposite_labels_for_different_requirements(self):
        scene = self.module.scene_for("train", 0, "scope_short", random.Random(12))
        rows = self.module.local_rows(scene, ByteBPE([]))
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual({json.loads(r["target"])["status"] for r in rows}, {"atendido", "nao_atendido"})
        self.assertEqual(rows[0]["units"], rows[1]["units"])
        self.audit_oracle(scene, rows)

    def test_long_document_local_labels_reconstruct_cross_batch_conflict(self):
        for seed in range(20):
            scene = self.module.scene_for("train", seed, "tail_conflict", random.Random(seed))
            rows = self.module.local_rows(scene, ByteBPE([]))
            if rows is not None:
                break
        self.assertIsNotNone(rows)
        self.assertGreater(len(rows), 4)
        self.assertTrue(any(not json.loads(r["target"])["evidence"] for r in rows))
        self.audit_oracle(scene, rows)

    def test_partial_evidence_at_boundary_is_not_given_a_label(self):
        scene = self.module.scene_for("train", 0, "tail_conflict", random.Random(5))
        text = scene["document"]["units"][0]["text"]
        scene["annotations"] = [dict(feature=scene["features"][0], status="atendido", start=0, end=len(text), quote=text)]
        self.assertIsNone(self.module.local_rows(scene, ByteBPE([])))
