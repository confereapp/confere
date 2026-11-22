import unittest
import torch
from confere_ai.distillation import correct_teacher_kl


class DistillationTests(unittest.TestCase):
    def test_wrong_teacher_tokens_and_prompt_are_not_anchored(self):
        logits = torch.zeros((1,3,2), requires_grad=True)
        teacher = torch.tensor([[[4.,0.],[4.,0.],[4.,0.]]], requires_grad=True)
        loss = correct_teacher_kl(logits, teacher, torch.tensor([[0,1,-100]])); loss.backward()
        self.assertGreater(logits.grad[0,0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0,1:].abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_no_correct_teacher_tokens_is_finite_zero(self):
        logits = torch.zeros((1,1,2), requires_grad=True)
        loss = correct_teacher_kl(logits, torch.tensor([[[4.,0.]]]), torch.tensor([[1]]))
        self.assertEqual(loss.item(),0);loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(),0)
