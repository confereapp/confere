import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from calibrate_evidence import evaluate_thresholds


class CalibrationTests(unittest.TestCase):
    def test_weak_prediction_is_filtered_without_copying_expected_output(self):
        rows = []
        for i, status in enumerate(["evidencia_insuficiente", "atendido"]):
            rows.append(dict(id=str(i), task="grounded_"+status, prompt="query", document="source.txt", grounding_schema=2,
                             units=[dict(id="page:1", text="source")], target=json.dumps(dict(status=status,
                             evidence=[] if i == 0 else [dict(status=status,unit="page:1")]))))
        probabilities = [[.7,.1],[.9,.1]]
        self.assertEqual(evaluate_thresholds(rows,probabilities,[.8,.5])["correct"],2)
        self.assertEqual(evaluate_thresholds(rows,probabilities,[.5,.5])["correct"],1)
        self.assertEqual(evaluate_thresholds(rows,[[.1,.1],[.1,.1]],[.8,.5])["correct"],1)
