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

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_boundary_has_finite_gradients_and_hook_is_disabled_for_inference(self):
        torch.manual_seed(42)
        model=GPT(GPTConfig(vocab_size=32,block_size=256,width=16,layers=3,heads=2)).to("mps")
        configure_trainable(model,1)
        seen=[]
        model.blocks[2].norm1.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].requires_grad))
        frozen={name:p.detach().cpu().clone() for name,p in model.named_parameters() if not p.requires_grad}
        for length in [128,256,128]:
            model.zero_grad(set_to_none=True)
            model(torch.randint(0,32,(2,length),device="mps")).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
            self.assertTrue(seen[-1])
        with torch.no_grad(): model(torch.tensor([[1,2,3]],device="mps"))
        self.assertFalse(seen[-1])
        self.assertTrue(all(torch.equal(dict(model.named_parameters())[name].detach().cpu(),value) for name,value in frozen.items()))
        configure_trainable(model)
        self.assertFalse(hasattr(model,"_frozen_input_hook"))
