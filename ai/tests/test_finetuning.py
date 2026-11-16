import unittest
import torch
from confere_ai.finetuning import configure_trainable
from confere_ai.model import GPT, GPTConfig


class FinetuningTests(unittest.TestCase):
    def test_frozen_embeddings_and_early_blocks_remain_bitwise_unchanged(self):
        torch.manual_seed(42)
        model = GPT(GPTConfig(vocab_size=32,block_size=16,width=16,layers=3,heads=2))
        count = configure_trainable(model,1)
        self.assertLess(count,sum(p.numel() for p in model.parameters()))
        before = {name:p.detach().clone() for name,p in model.named_parameters()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01)
        model(torch.tensor([[1,2,3]])).square().mean().backward()
        self.assertTrue(all(p.grad is None for name,p in model.named_parameters() if name.startswith(("token_embedding", "position_embedding", "blocks.0.", "blocks.1."))))
        optimizer.step()
        for name,p in model.named_parameters():
            if not p.requires_grad: self.assertTrue(torch.equal(p,before[name]))
        self.assertTrue(any(not torch.equal(p,before[name]) for name,p in model.named_parameters() if p.requires_grad))
        self.assertEqual(configure_trainable(model),sum(p.numel() for p in model.parameters()))

    def test_invalid_selection_does_not_mutate_flags(self):
        model=GPT(GPTConfig(vocab_size=32,block_size=16,width=16,layers=2,heads=2))
        for invalid in [0,3,-1]:
            with self.assertRaises(ValueError): configure_trainable(model,invalid)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
