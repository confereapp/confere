import unittest
import torch
from confere_ai.model import GPT,GPTConfig
from confere_ai.finetuning import configure_trainable


class NormBiasTuningTests(unittest.TestCase):
    def test_only_final_bias_changes_and_logits_shift_equally_at_each_position(self):
        torch.manual_seed(42)
        model=GPT(GPTConfig(vocab_size=260,block_size=32,width=16,layers=1,heads=2))
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.assertEqual(configure_trainable(model,norm_bias_only=True),16)
        ids=torch.tensor([[256,65,258]])
        original=model(ids).detach();loss=torch.nn.functional.cross_entropy(model(ids).flatten(0,1),torch.tensor([83,83,83]))
        loss.backward();optimizer=torch.optim.SGD(model.parameters(),lr=.1);optimizer.step()
        for name,value in model.state_dict().items():
            if name != 'norm.bias':self.assertTrue(torch.equal(value,before[name]))
        self.assertFalse(torch.equal(model.norm.bias,before['norm.bias']))
        shift=model(ids).detach()-original
        self.assertTrue(torch.allclose(shift[:,0],shift[:,1],atol=1e-6))
        with self.assertRaises(ValueError):configure_trainable(model,1,norm_bias_only=True)
