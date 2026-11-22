import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_output_prior import fit_prior


class OutputPriorTests(unittest.TestCase):
    def test_training_fit_changes_only_an_additive_prior(self):
        logits=torch.tensor([[0.,1.],[4.,-4.],[-5.,5.]])
        weights=torch.eye(2);before=logits.clone();labels=torch.tensor([0,0,1])
        delta,stats=fit_prior(logits,weights,labels)
        self.assertLess(stats['final_loss'],stats['initial_loss'])
        self.assertTrue(torch.equal(logits,before))
        self.assertTrue(torch.equal((logits+delta@weights.T).argmax(-1),labels))
