import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_conversation_routing import RoutedDecoder,FusionDecoder,presence_pools,TypeSelectedDecoder,evaluation_decoder
from confere_ai.evidence_classifier import paired_indices
from confere_ai.tokenizer import ByteBPE


class ConversationRoutingTests(unittest.TestCase):
    def test_evaluation_uses_the_same_selected_decoder_pipeline(self):
        tokenizer=ByteBPE([]);state=[0]
        def classifier(ids,lengths):
            logits=torch.full((1,6),-20.);logits[0,state[0]]=20.;return logits
        class Decoder:
            config=1
            def __init__(self,text):self.text=text
            def generate(self,prompt,budget,eos):return torch.cat([prompt,torch.tensor([tokenizer.encode(self.text)+[eos]])],-1)
            def __call__(self,ids):return torch.zeros(len(ids),ids.shape[1],tokenizer.vocab_size)
        legacy,factual=Decoder('Olá.'),Decoder('42 dias.')
        model=evaluation_decoder(legacy,classifier,tokenizer,2.,fusion=True,length_normalized=True,factual=factual)
        prompt=torch.tensor([[1,2]])
        self.assertEqual(tokenizer.decode(model.generate(prompt,256,tokenizer.EOS)[0,2:].tolist()),'42 dias.')
        state[0]=5
        self.assertEqual(tokenizer.decode(model.generate(prompt,256,tokenizer.EOS)[0,2:].tolist()),'Olá.')
        with self.assertRaises(ValueError):evaluation_decoder(legacy,classifier,tokenizer,2.,fusion=False,length_normalized=True,factual=factual)

    def test_learned_type_selects_factual_decoder_and_preserves_other_types(self):
        state=[0]
        def classifier(prompt,lengths):
            result=torch.zeros(1,6);result[0,state[0]]=1;return result
        class Adapter:
            config=object();tokenizer=object()
            def __init__(self,value):self.value=value;self.classifier=classifier
            def generate(self,prompt,budget,eos):return self.value,budget,eos
        factual,legacy=Adapter('factual'),Adapter('legacy')
        model=TypeSelectedDecoder(factual,legacy);prompt=torch.tensor([[1,2]])
        self.assertEqual(model.generate(prompt,16,3),('factual',16,3))
        for i in range(1,6):
            state[0]=i
            self.assertEqual(model.generate(prompt,16,3),('legacy',16,3))
        legacy.classifier=lambda *args:None
        with self.assertRaises(ValueError):TypeSelectedDecoder(factual,legacy)

    def test_paired_sampling_changes_the_answer_for_the_same_training_body(self):
        rows=[dict(split='train',target=target,scenario=dict(paired_presence_group=group,text='body-'+group))
              for group in ['a','b'] for target in ['Sim.','Não.','Não tenho informação suficiente para responder.']]
        pools=presence_pools(rows)
        indices=paired_indices(rows,pools,8,torch.Generator().manual_seed(3),label_key='target')
        for a,b in zip(indices[::2],indices[1::2]):
            self.assertEqual(rows[a]['scenario'],rows[b]['scenario'])
            self.assertNotEqual(rows[a]['target'],rows[b]['target'])
        rows[0]['split']='validation'
        with self.assertRaises(ValueError):presence_pools(rows)
        rows[0]['split']='train';rows[0]['scenario']['text']='outro corpo'
        with self.assertRaises(ValueError):presence_pools(rows)

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
