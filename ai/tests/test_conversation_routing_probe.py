import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_conversation_routing import RoutedDecoder,FusionDecoder
from confere_ai.tokenizer import ByteBPE


class ConversationRoutingTests(unittest.TestCase):
    def test_fusion_scores_candidates_with_language_and_learned_state_probabilities(self):
        tokenizer=ByteBPE([])
        class Decoder:
            config=object()
            def generate(self,prompt,budget,eos):
                return torch.cat([prompt,torch.tensor([tokenizer.encode('Não tenho informação suficiente para responder.')+[eos]])],-1)
            def __call__(self,ids):return torch.zeros(len(ids),ids.shape[1],tokenizer.vocab_size)
        def classifier(ids,lengths):
            logits=torch.full((1,6),-10.);logits[0,3]=10.;return logits
        prompt=torch.tensor([[1,2]])
        for normalized in [False,True]:
            model=FusionDecoder(Decoder(),classifier,tokenizer,1.);model.length_normalized=normalized
            result=model.generate(prompt,256,tokenizer.EOS)
            self.assertEqual(tokenizer.decode(result[0,2:].tolist()),'Sim.')

    def test_learned_states_are_verbalized_and_facts_delegate_to_decoder(self):
        class Decoder:
            config=object()
            def generate(self,prompt,budget,eos):return torch.cat([prompt,torch.tensor([[eos]])],-1)
        tokenizer=ByteBPE([]);prompt=torch.tensor([[1,2]])
        def prediction(state):
            logits=torch.full((1,6),-10.);logits[0,state]=10.
            return lambda ids,lengths:logits
        router=RoutedDecoder(Decoder(),prediction(2),tokenizer,.95)
        answer=router.generate(prompt,256,tokenizer.EOS)
        self.assertEqual(tokenizer.decode(answer[0,2:].tolist()),'Há informações conflitantes no texto.')
        for state in [0,5]:
            router.classifier=prediction(state)
            self.assertEqual(router.generate(prompt,256,tokenizer.EOS).shape[-1],3)
        router.classifier=lambda ids,lengths:torch.zeros(1,6)
        self.assertEqual(router.generate(prompt,256,tokenizer.EOS).shape[-1],3)
