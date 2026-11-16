import json
import unittest
import torch
from confere_ai.evidence_classifier import EvidenceClassifier, encode_unit, response
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


class EvidenceClassifierTests(unittest.TestCase):
    def test_predicted_states_preserve_both_conflicts_and_never_invent_references(self):
        units = [dict(id="page:2"), dict(id="page:4")]
        result = json.loads(response(units, [[.9, .1], [.1, .9]]))
        self.assertEqual(result["status"], "evidencia_insuficiente")
        self.assertEqual({e["unit"] for e in result["evidence"]}, {"page:2", "page:4"})
        self.assertEqual(json.loads(response(units, [[.1, .1], [.1, .1]]))["evidence"], [])
        self.assertEqual(len(json.loads(response(units[:1], [[.9, .9]]))["evidence"]), 2)
        for values in [[], [[float("nan"), .1], [.1, .1]], [[2., .1], [.1, .1]]]:
            with self.assertRaises(ValueError): response(units, values)

    def test_readout_ignores_future_padding_and_context_overflow_is_rejected(self):
        torch.manual_seed(42)
        model = EvidenceClassifier(GPT(GPTConfig(vocab_size=260, block_size=32, width=16, layers=2, heads=2))).eval()
        first = model(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([3]))
        second = model(torch.tensor([[1, 2, 3, 9, 9]]), torch.tensor([3]))
        self.assertTrue(torch.allclose(first, second))
        first.sum().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        with self.assertRaises(ValueError): encode_unit("testes", "x" * 100, ByteBPE(), 32)
