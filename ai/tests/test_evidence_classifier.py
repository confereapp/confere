import json
import sys
import unittest
from pathlib import Path
import torch
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from confere_ai.evidence_classifier import EvidenceClassifier, EvidencePredictor, encode_unit, response, decision_prefix, fold_normalization
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


class EvidenceClassifierTests(unittest.TestCase):
    def test_sentence_predictions_preserve_original_alias_and_opposing_evidence(self):
        from dataclasses import asdict
        from confere_ai.evidence_classifier import sentence_parts
        config=GPTConfig(vocab_size=260,block_size=512,width=16,layers=1,heads=2)
        model=EvidenceClassifier(GPT(config))
        ckpt=dict(config=asdict(config),merges=[],classifier=model.state_dict(),settings=dict(objective='document_classifier',readout='shared_json_decision_prefix_v1',unit_segmentation='sentences_v1'))
        predictor=EvidencePredictor(ckpt,'cpu')
        units=[dict(id='original:4',text='Há testes. Não há testes.'),dict(id='original:8',text='Outra informação.')]
        with patch.object(predictor.model,'forward',return_value=torch.tensor([[9.,-9.],[-9.,9.],[-9.,-9.]])):
            output=json.loads(predictor.predict_units('Testes',units))
        self.assertEqual(output['status'],'evidencia_insuficiente')
        self.assertEqual({e['unit'] for e in output['evidence']},{'original:4'})
        self.assertEqual(len(output['evidence']),2)
        with patch.object(predictor.model,'forward',return_value=torch.tensor([[9.,-9.],[float('nan'),9.],[-9.,-9.]])):
            with self.assertRaises(ValueError): predictor.predict_units('Testes',units)
        text='Uma frase!  Outra.\nÚltima🙂'
        self.assertEqual(''.join(sentence_parts(text)),text)
        with self.assertRaises(ValueError): EvidencePredictor(dict(ckpt,settings=dict(ckpt['settings'],unit_segmentation='invalid')),'cpu')

    def test_predictor_uses_provided_units_without_parsing_or_inventing_aliases(self):
        from dataclasses import asdict
        config = GPTConfig(vocab_size=260, block_size=512, width=16, layers=1, heads=2)
        model = EvidenceClassifier(GPT(config))
        checkpoint = dict(config=asdict(config), merges=[], classifier=model.state_dict(),
                          settings=dict(objective="document_classifier", readout="shared_json_decision_prefix_v1"))
        predictor = EvidencePredictor(checkpoint, "cpu")
        units = [dict(id="page:2", text="Há testes."), dict(id="page:4", text="Não há testes.")]
        with patch.object(predictor.model, "forward", return_value=torch.tensor([[9., -9.], [-9., 9.]])):
            result = json.loads(predictor.predict_units("Testes", units))
        self.assertEqual({e["unit"] for e in result["evidence"]}, {"page:2", "page:4"})
        self.assertEqual(result["status"], "evidencia_insuficiente")
        with self.assertRaises(ValueError): predictor.predict_units("Testes", units * 2)
        with self.assertRaises(ValueError): EvidencePredictor(dict(checkpoint, settings={}), "cpu")
    def test_paired_sampling_uses_same_source_and_opposing_targets_when_available(self):
        from confere_ai.evidence_classifier import paired_indices
        items = [dict(source_key="a", labels=[1., 0.]), dict(source_key="a", labels=[0., 1.]),
                 dict(source_key="a", labels=[1., 0.]), dict(source_key="b", labels=[0., 0.])]
        pools = [[0, 1, 2], [3]]; generator = torch.Generator().manual_seed(42)
        for _ in range(30):
            indices = paired_indices(items, pools, 8, generator)
            for a, b in zip(indices[::2], indices[1::2]):
                self.assertEqual(items[a]["source_key"], items[b]["source_key"])
                if items[a]["source_key"] == "a": self.assertNotEqual(items[a]["labels"], items[b]["labels"])
        with self.assertRaises(ValueError): paired_indices(items, pools, 3, generator)

    def test_suffix_query_is_after_source_and_training_matches_inference_encoding(self):
        tokenizer = ByteBPE([])
        readout = "suffix_query_decision_prefix_v1"
        ids = encode_unit("Incluir backup.", "Há testes.", tokenizer, 512, readout=readout)
        text = tokenizer.decode(ids)
        self.assertLess(text.index("Há testes."), text.index("Incluir backup."))
        self.assertNotIn("atendido", text)
        from confere_ai.evidence_classifier import examples
        row = dict(id="example", requirement="Incluir backup.", units=[dict(id="page:4", text="Há testes.")],
                   grounding_schema=2, target=json.dumps(dict(status="evidencia_insuficiente", evidence=[])))
        self.assertEqual(examples([row], tokenizer, 512, readout=readout)[0]["ids"], ids)
        with self.assertRaises(ValueError): encode_unit("Backup", "Texto", tokenizer, 512, readout="unknown")

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

    def test_calibrated_thresholds_filter_weak_predictions_without_changing_references(self):
        units = [dict(id="page:4")]
        self.assertEqual(json.loads(response(units, [[.7, .55]], [.8, .6]))["evidence"], [])
        self.assertEqual(json.loads(response(units, [[.9, .55]], [.8, .6]))["evidence"], [dict(status="atendido", unit="page:4")])
        for invalid in [[.5], [float("nan"), .5], [1., .5]]:
            with self.assertRaises(ValueError): response(units, [[.9,.1]], invalid)

    def test_four_states_preserve_initial_binary_decisions_and_learn_conflict_jointly(self):
        from confere_ai.evidence_classifier import load_binary_as_categorical
        config = GPTConfig(vocab_size=260, block_size=32, width=16, layers=1, heads=2)
        old = EvidenceClassifier(GPT(config)); new = EvidenceClassifier(GPT(config), "categorical_four_state")
        load_binary_as_categorical(new, old.state_dict())
        features = torch.randn(50,16)
        self.assertTrue(torch.equal((old.head(features).sigmoid() >= .5).float(), new.evidence_scores(new.head(features))))
        logits = torch.tensor([[4.,0.,0.,0.],[0.,4.,0.,0.],[0.,0.,4.,0.],[0.,0.,0.,4.]], requires_grad=True)
        labels = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        self.assertTrue(torch.equal(new.evidence_scores(logits), labels))
        new.loss(logits,labels).backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_predictor_loads_categorical_state_metadata_and_keeps_aliases(self):
        from dataclasses import asdict
        config = GPTConfig(vocab_size=260, block_size=512, width=16, layers=1, heads=2)
        model = EvidenceClassifier(GPT(config), "categorical_four_state")
        checkpoint = dict(config=asdict(config),merges=[],classifier=model.state_dict(),settings=dict(objective="document_classifier",readout="shared_json_decision_prefix_v1",state_mode="categorical_four_state"))
        predictor = EvidencePredictor(checkpoint,"cpu")
        units = [dict(id="page:2",text="Há testes."),dict(id="page:4",text="Não há testes.")]
        with patch.object(predictor.model,"forward",return_value=torch.tensor([[0.,9.,0.,0.],[0.,0.,9.,0.]])):
            result = json.loads(predictor.predict_units("Testes",units))
        self.assertEqual(result["status"],"evidencia_insuficiente")
        self.assertEqual({e["unit"] for e in result["evidence"]},{"page:2","page:4"})

    def test_pair_sampler_supports_conversation_targets_without_changing_context(self):
        from confere_ai.evidence_classifier import paired_indices
        rows = [dict(target="Sim."),dict(target="Não."),dict(target="Sim.")]
        indices = paired_indices(rows, [[0,1,2]], 8, torch.Generator().manual_seed(42), label_key="target")
        self.assertTrue(all(rows[a]["target"] != rows[b]["target"] for a,b in zip(indices[::2],indices[1::2])))
