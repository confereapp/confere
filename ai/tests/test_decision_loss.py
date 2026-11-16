import unittest
import torch
from torch.nn import functional as F
from confere_ai.decision_loss import decision_weights, weighted_loss, weight_batch
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample


class DecisionLossTests(unittest.TestCase):
    def test_utf8_and_merged_tokens_align_with_supervised_values_only(self):
        row = dict(id="probe", task="grounded_atendido", prompt="Há documentação?",
                   target='{"status": "atendido", "evidence": [{"status": "atendido", "unit": "page:2"}]}')
        tokenizer = ByteBPE.train([row["target"]] * 4, 300)
        _, labels = encode_sample(row, tokenizer, 1024)
        weights = decision_weights(row, tokenizer, labels, 4.)
        self.assertEqual(len(weights), len(labels))
        self.assertTrue(all(weight == 0 for label, weight in zip(labels, weights) if label == -100))
        self.assertEqual(weights[-1], 1.)  # EOS mantém supervisão normal.
        self.assertIn(4., weights)
        for label, weight in zip(labels, weights):
            if weight == 4.:
                self.assertNotEqual(label, -100)
        bad = labels.copy(); bad[-2] = tokenizer.PAD
        with self.assertRaises(ValueError): decision_weights(row, tokenizer, bad, 4.)
        row["task"] = "conversation_identity"
        self.assertNotIn(4., decision_weights(row, tokenizer, labels, 4.))

    def test_weighted_mean_and_gradients_ignore_padding_and_default_matches_ce(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 8, requires_grad=True)
        labels = torch.tensor([[-100, 2, 3, 4], [1, 2, -100, -100]])
        weights = weight_batch([[0., 4., 1., 1.], [1., 4.]], [0, 1], labels.shape, "cpu")
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
        self.assertTrue(torch.allclose(weighted_loss(logits, labels, weights), (losses * weights).sum() / 11.))
        weighted_loss(logits, labels, weights).backward()
        self.assertTrue(torch.equal(logits.grad[labels == -100], torch.zeros_like(logits.grad[labels == -100])))
        self.assertTrue(torch.allclose(weighted_loss(logits, labels, (labels != -100).float()),
                                       F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)))
