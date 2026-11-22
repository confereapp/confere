import unittest
import torch
from confere_ai.decision_loss import example_mean_loss


class FirstTokenWeightTests(unittest.TestCase):
    def test_weight_targets_first_response_token_without_counting_prompt_or_padding(self):
        logits=torch.zeros(1,5,3,requires_grad=True)
        labels=torch.tensor([[-100,-100,1,2,-100]])
        example_mean_loss(logits,labels,8.).backward()
        gradients=logits.grad.abs().sum(-1)[0]
        self.assertAlmostEqual((gradients[2]/gradients[3]).item(),8.,places=5)
        self.assertEqual(gradients[:2].sum().item()+gradients[-1].item(),0.)
        for value in [0.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):example_mean_loss(logits,labels,value)


class ExampleLossTests(unittest.TestCase):
    def test_short_answers_receive_equal_example_weight(self):
        logits = torch.zeros((2,3,2), requires_grad=True)
        labels = torch.tensor([[1,-100,-100],[1,1,1]])
        example_mean_loss(logits, labels).backward()
        self.assertAlmostEqual(logits.grad[0].abs().sum().item(), logits.grad[1].abs().sum().item(), places=6)
        self.assertEqual(logits.grad[0,1:].abs().sum().item(),0)
        with self.assertRaises(ValueError): example_mean_loss(logits, torch.full_like(labels,-100))
