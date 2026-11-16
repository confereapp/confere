import unittest
import torch
from confere_ai.decision_loss import example_mean_loss


class ExampleLossTests(unittest.TestCase):
    def test_short_answers_receive_equal_example_weight(self):
        logits = torch.zeros((2,3,2), requires_grad=True)
        labels = torch.tensor([[1,-100,-100],[1,1,1]])
        example_mean_loss(logits, labels).backward()
        self.assertAlmostEqual(logits.grad[0].abs().sum().item(), logits.grad[1].abs().sum().item(), places=6)
        self.assertEqual(logits.grad[0,1:].abs().sum().item(),0)
        with self.assertRaises(ValueError): example_mean_loss(logits, torch.full_like(labels,-100))
