import unittest
import torch
from confere_ai.model import GPT,GPTConfig
from confere_ai.semantic_auxiliary import semantic_class,semantic_forward


class SemanticAuxiliaryTests(unittest.TestCase):
    def test_classes_and_causal_features_do_not_read_the_gold_response(self):
        torch.manual_seed(42)
        model=GPT(GPTConfig(vocab_size=20,block_size=16,width=16,layers=1,heads=2))
        head=torch.nn.Linear(16,6)
        ids=torch.tensor([[1,2,3,4,5]]);labels=torch.tensor([[-100,-100,7,8,9]])
        classes=torch.tensor([2])
        logits,loss=semantic_forward(model,ids,labels,head,classes)
        changed=ids.clone();changed[:,3:]=19
        _,other=semantic_forward(model,changed,labels,head,classes)
        self.assertTrue(torch.allclose(loss,other))
        loss.backward();self.assertIsNotNone(head.weight.grad)
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())
        self.assertEqual(len(model.norm._forward_hooks),0)
        self.assertEqual(semantic_class(dict(task='conversation_context_answer_conflict',target='Há informações conflitantes no texto.')),2)
        self.assertEqual(semantic_class(dict(task='conversation_context_answer_fact',target='18 dias.')),0)
        self.assertEqual(semantic_class(dict(task='conversation_identity',target='Sou o Confere.')),5)
        self.assertEqual(logits.shape,(1,5,20))
