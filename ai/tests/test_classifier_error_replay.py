import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_evidence_classifier import training_errors


class ClassifierErrorReplayTests(unittest.TestCase):
    def test_mines_both_missing_and_spurious_evidence_from_supplied_training_units(self):
        class Model:
            backbone=type('Backbone',(),{'config':type('Config',(),{'block_size':128})()})()
            def eval(self): return self
            def __call__(self,ids,lengths): return torch.tensor([[.8,.1],[.1,.8],[.1,.1]])
            def evidence_scores(self,logits): return logits
        rows=[dict(ids=[1,2],labels=[1.,0.]),dict(ids=[1,3],labels=[0.,0.]),dict(ids=[1,4],labels=[1.,0.])]
        self.assertEqual(training_errors(Model(),rows,'cpu'),[1,2])

    def test_mining_uses_the_sources_calibrated_thresholds(self):
        class Model:
            backbone=type('Backbone',(),{'config':type('Config',(),{'block_size':128})()})()
            def eval(self):return self
            def __call__(self,ids,lengths):return torch.tensor([[.4,.2]])
            def evidence_scores(self,logits):return logits
        rows=[dict(ids=[1,2],labels=[1.,1.])]
        self.assertEqual(training_errors(Model(),rows,'cpu'),[0])
        self.assertEqual(training_errors(Model(),rows,'cpu',[.3,.1]),[])
        with self.assertRaises(ValueError):training_errors(Model(),rows,'cpu',[float('nan'),.1])
