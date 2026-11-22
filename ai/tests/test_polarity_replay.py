import importlib.util
import json
import sys
import unittest
from pathlib import Path

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('polarity_replay',scripts/'prepare_polarity_replay.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PolarityReplayTests(unittest.TestCase):
    def test_attestation_is_present_in_both_polarities_without_copying_diagnostic_frame(self):
        pairs=[pair for pair in builder.FORMS if 'atesta' in pair[0]]
        self.assertEqual(len(pairs),4)
        for positive,negative in pairs:
            self.assertIn('atesta',negative)
            self.assertNotIn('atesta que {} está disponível',positive)
            rows=builder.documents(80,topics=['cadastro de itens','consulta de contas'],forms=[(positive,negative)],sentence_units=True)
            states=set()
            for row in rows:
                target=json.loads(row['target'])
                states.update(e['status'] for e in target['evidence'])
            self.assertEqual(states,{'atendido','nao_atendido'})
