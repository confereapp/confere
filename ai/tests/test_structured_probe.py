import json
import unittest
import torch
from confere_ai.structured_probe import candidates,choose
from confere_ai.grounding import resolve
from confere_ai.tokenizer import ByteBPE
from confere_ai.model import GPT,GPTConfig


class StructuredProbeTests(unittest.TestCase):
    def test_all_candidates_have_valid_schema_and_only_typed_aliases(self):
        aliases=['page:1','page:2']
        values=list(candidates(aliases))
        self.assertEqual(len(values),16)
        for value in values:
            resolved=resolve(value,'document',[dict(id=a,text='texto') for a in aliases],2)
            self.assertTrue(resolved['valid_format'] and resolved['reference_valid'])
        with self.assertRaises(ValueError):list(candidates(['page:99']))

    def test_batched_probability_matches_individual_scoring_and_context_is_guarded(self):
        torch.manual_seed(42)
        tokenizer=ByteBPE()
        model=GPT(GPTConfig(vocab_size=tokenizer.vocab_size,block_size=256,width=16,layers=1,heads=2)).eval()
        prefix=[tokenizer.BOS,*tokenizer.encode('pergunta'),tokenizer.SEP]
        single=choose(model,tokenizer,prefix,['page:1'],device='cpu',batch_size=1)
        self.assertEqual(single,choose(model,tokenizer,prefix,['page:1'],device='cpu',batch_size=8))
        self.assertIn(single,list(candidates(['page:1'])))
        with self.assertRaises(ValueError):choose(model,tokenizer,prefix,['page:1'],device='cpu',response_tokens=1)
