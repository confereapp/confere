import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict

import torch
from confere_ai.evidence_classifier import EvidenceClassifier, EvidencePredictor
from confere_ai.model import GPT, GPTConfig
from probe_conditioned_evidence import FrozenPairs, ConditionalHead, ConditionedPredictor, load_probe, OBJECTIVE


class ConditionedEvidenceTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        config=GPTConfig(vocab_size=260,block_size=512,width=16,layers=1,heads=2)
        model=EvidenceClassifier(GPT(config))
        source=dict(config=asdict(config),merges=[],classifier=model.state_dict(),step=0,settings=dict(
            objective='document_classifier',readout='shared_json_decision_prefix_v1',unit_segmentation='sentences_v1'))
        encoder=FrozenPairs(source,'cpu')
        head=ConditionalHead(torch.zeros(98),torch.ones(98))
        return source,encoder,head

    def test_zero_residual_matches_native_and_head_training_keeps_encoder_frozen(self):
        source,encoder,head=self.fixture();predictor=ConditionedPredictor(encoder,head)
        query='Incluir backup.';units=[dict(id='original:7',text='Há backup. Não há backup.'),dict(id='original:8',text='Outro assunto.')]
        self.assertEqual(predictor.predict_units(query,units),encoder.base.predict_units(query,units))
        x,b=encoder.features([(query,'Há backup.')])
        optimizer=torch.optim.SGD(head.parameters(),lr=.1)
        head(x,b).sum().backward();optimizer.step()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.base.model.parameters()))
        with self.assertRaises(ValueError):predictor.predict_units('',units)
        with self.assertRaises(ValueError):predictor.predict_units(query,[units[0],units[0]])

    def test_probe_roundtrip_verifies_source_hash_and_stays_out_of_native_loader(self):
        source,encoder,head=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);torch.save(source,root/'source.pt')
            state=dict(settings=dict(objective=OBJECTIVE,promotable=False),head=head.state_dict(),step=0,
                       source_sha256=hashlib.sha256((root/'source.pt').read_bytes()).hexdigest())
            path=root/'probe.pt';torch.save(state,path)
            loaded=load_probe(path,'cpu');units=[dict(id='page:1',text='Há backup.')]
            self.assertEqual(loaded.predict_units('Backup',units),encoder.base.predict_units('Backup',units))
            with self.assertRaises(ValueError):EvidencePredictor(state,'cpu')
            state['source_sha256']='0'*64;torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'Origem congelada'):load_probe(path,'cpu')

    def test_smaller_linear_head_zero_start_roundtrip_and_unknown_kind_rejection(self):
        source,encoder,mlp=self.fixture()
        head=ConditionalHead(torch.zeros(98),torch.ones(98),'linear_residual_v1')
        self.assertLess(sum(p.numel() for p in head.parameters()),sum(p.numel() for p in mlp.parameters()))
        x=torch.randn(2,98);base=torch.randn(2,2)
        self.assertTrue(torch.equal(head(x,base),base))
        with torch.no_grad():head.net.weight.fill_(.02);head.net.bias.fill_(-.03)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);torch.save(source,root/'source.pt')
            state=dict(settings=dict(objective=OBJECTIVE,head_kind='linear_residual_v1'),head=head.state_dict(),
                       source_sha256=hashlib.sha256((root/'source.pt').read_bytes()).hexdigest())
            torch.save(state,root/'probe.pt');loaded=load_probe(root/'probe.pt','cpu')
            self.assertTrue(torch.equal(head(x,base),loaded.head(x,base)))
            state['settings']['head_kind']='unknown';torch.save(state,root/'probe.pt')
            with self.assertRaisesRegex(ValueError,'Cabeça residual'):load_probe(root/'probe.pt','cpu')


if __name__=='__main__':unittest.main()
