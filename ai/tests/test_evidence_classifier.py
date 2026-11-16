import json
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from confere_ai.evidence_classifier import EvidenceClassifier, encode_unit, response, decision_prefix, fold_normalization
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


class EvidenceClassifierTests(unittest.TestCase):
    def test_normalization_folding_preserves_logits_without_inference_preprocessing(self):
        torch.manual_seed(42)
        head = torch.nn.Linear(4, 2).double(); raw = torch.nn.Linear(4, 2)
        mean = torch.randn(4).double(); scale = torch.rand(4).double() + .1
        x = torch.randn(7, 4)
        fold_normalization(head, raw, mean, scale)
        self.assertTrue(torch.allclose(raw(x).double(), head((x.double() - mean) / scale), atol=1e-6))
        with self.assertRaises(ValueError): fold_normalization(head, raw, mean, torch.zeros(4))
    def test_readout_prefix_contains_no_class_dependent_tokens(self):
        tokenizer = ByteBPE.train([json.dumps(dict(status=s, evidence=[])) for s in
                                   ("atendido", "nao_atendido", "evidencia_insuficiente")] * 4, 300)
        prefix = decision_prefix(tokenizer)
        next_tokens = []
        for status in ("atendido", "nao_atendido", "evidencia_insuficiente"):
            ids = tokenizer.encode(json.dumps(dict(status=status, evidence=[])))
            self.assertEqual(ids[:len(prefix)], prefix)
            next_tokens.append(ids[len(prefix)])
        self.assertGreater(len(set(next_tokens)), 1)
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

    def test_head_evaluation_uses_predictions_instead_of_copying_gold_evidence(self):
        from probe_evidence_head import evaluate_head
        row = dict(id="probe", task="grounded_evidencia_insuficiente", prompt="probe", document="author.txt", grounding_schema=2,
                   units=[dict(id="page:2", text="Há testes."), dict(id="page:4", text="Não há testes.")],
                   target=json.dumps(dict(status="evidencia_insuficiente", evidence=[dict(status="atendido", unit="page:2"), dict(status="nao_atendido", unit="page:4")])))
        head = torch.nn.Identity()
        self.assertEqual(evaluate_head(head, torch.tensor([[5., -5.], [-5., 5.]]), [row])["correct"], 1)
        self.assertEqual(evaluate_head(head, torch.full((2, 2), -5.), [row])["correct"], 0)
