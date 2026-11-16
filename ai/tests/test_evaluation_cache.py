import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from confere_ai.evaluation_cache import protocol_fingerprint, reuse_baseline


class EvaluationCacheTests(unittest.TestCase):
    def test_exact_reuse_rejects_history_weights_and_protocol_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "best.pt"; checkpoint.write_bytes(b"own-checkpoint")
            rows = [dict(id=split, project_id=split, split=split, task="conversation_test", history=[], turn_index=0,
                         response_tokens=256, prompt="oi-"+split, target="olá", provenance=dict(kind="synthetic_authored")) for split in ["train", "validation"]]
            corpus = root / "data.jsonl"; corpus.write_text("\n".join(json.dumps(r) for r in rows))
            info = dict(evaluation_protocol_hash=protocol_fingerprint(), paired_evaluation_context=1024, paired_evaluation_response_tokens=256,
                        data=str(corpus), dataset_fingerprint=hashlib.sha256(corpus.read_bytes()).hexdigest())
            audit = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), selected_step=1)
            selected = dict(total=1, correct=1, samples=[dict(id="validation", task="conversation_test", prompt="oi-validation", expected="olá", actual="olá", exact_match=True)])
            for name, value in [("run.json",info),("context-audit.json",audit),("instruction-evaluation.json",dict(selected=selected, selected_step=1,test_examples_used=0))]:
                (root/name).write_text(json.dumps(value))
            self.assertEqual(reuse_baseline(root,checkpoint,[rows[1]],1024,256), selected)
            changed = copy.deepcopy(rows[1]); changed["history"] = [["before", "answer"]]
            with self.assertRaises(ValueError): reuse_baseline(root,checkpoint,[changed],1024,256)
            with self.assertRaises(ValueError): reuse_baseline(root,checkpoint,[rows[1]],1024,128)
            bad_info = {**info, "evaluation_protocol_hash": "changed-code"}
            (root/"run.json").write_text(json.dumps(bad_info))
            with self.assertRaises(ValueError): reuse_baseline(root,checkpoint,[rows[1]],1024,256)
            (root/"run.json").write_text(json.dumps(info))
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError): reuse_baseline(root,checkpoint,[rows[1]],1024,256)
