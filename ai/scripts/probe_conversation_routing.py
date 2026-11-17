"""Ensaio offline de classificação aprendida e decoder próprio; sem promoção."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from collections import defaultdict
from confere_ai.chat import prepare_prompt
from confere_ai.evidence_classifier import unit_batch,paired_indices
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import load_instructions,score,conversation_indices
from confere_ai.model import GPT,GPTConfig
from confere_ai.semantic_auxiliary import semantic_class
from confere_ai.tokenizer import ByteBPE

STATES={1:'Não tenho informação suficiente para responder.',2:'Há informações conflitantes no texto.',3:'Sim.',4:'Não.'}


def presence_pools(rows):
    grouped=defaultdict(list)
    for i,row in enumerate(rows):
        if row['split']!='train':raise ValueError('Consultas pareadas usam somente treino.')
        group=row.get('scenario',{}).get('paired_presence_group')
        if group is not None:grouped[group].append(i)
    if any(len({rows[i]['scenario'].get('text') for i in pool})!=1 or not rows[pool[0]]['scenario'].get('text') for pool in grouped.values()):
        raise ValueError('Grupo pareado precisa preservar o mesmo texto.')
    pools=[pool for pool in grouped.values() if len({rows[i]['target'] for i in pool})>1]
    if not pools:raise ValueError('Não há consultas pareadas com respostas distintas.')
    return pools


class Classifier(torch.nn.Module):
    def __init__(self,source,head):
        super().__init__();self.encoder=GPT(GPTConfig(**source['config']))
        self.encoder.load_state_dict(source['model']);self.encoder.head=torch.nn.Identity()
        self.head=torch.nn.Linear(source['config']['width'],6)
        if head is not None:self.head.load_state_dict(head)
    def forward(self,ids,lengths):
        return self.head(self.encoder(ids)[torch.arange(len(ids),device=ids.device),lengths-1])


class RoutedDecoder:
    def __init__(self,decoder,classifier,tokenizer,threshold):
        self.decoder=decoder;self.classifier=classifier;self.tokenizer=tokenizer
        self.threshold=threshold;self.config=decoder.config
    @torch.no_grad()
    def generate(self,prompt,max_new_tokens,eos):
        probabilities=self.classifier(prompt,torch.tensor([prompt.shape[1]],device=prompt.device)).softmax(-1)[0]
        confidence,state=probabilities.max(0);state=int(state)
        if state in STATES and confidence.item()>=self.threshold:
            answer=self.tokenizer.encode(STATES[state])+[eos]
            if len(answer)>max_new_tokens:raise ValueError('Verbalização excede o orçamento.')
            return torch.cat([prompt,torch.tensor([answer],device=prompt.device)],-1)
        return self.decoder.generate(prompt,max_new_tokens,eos)


class TypeSelectedDecoder:
    """Preserva o decoder legado e usa o factual conforme o tipo aprendido."""
    def __init__(self,factual,legacy):
        if (factual.config!=legacy.config or factual.classifier is not legacy.classifier
                or factual.tokenizer is not legacy.tokenizer):
            raise ValueError('Componentes de roteamento incompatíveis.')
        self.factual=factual;self.legacy=legacy;self.config=factual.config

    @torch.no_grad()
    def generate(self,prompt,max_new_tokens,eos):
        logits=self.factual.classifier(prompt,torch.tensor([prompt.shape[1]],device=prompt.device))
        decoder=self.factual if int(logits.argmax(-1).item())==0 else self.legacy
        return decoder.generate(prompt,max_new_tokens,eos)


class FusionDecoder(RoutedDecoder):
    @torch.no_grad()
    def generate(self,prompt,max_new_tokens,eos):
        probabilities=self.classifier(prompt,torch.tensor([prompt.shape[1]],device=prompt.device)).softmax(-1)[0]
        native=self.decoder.generate(prompt,max_new_tokens,eos)
        candidates=[native[0,prompt.shape[1]:].tolist()]+[self.tokenizer.encode(text)+[eos] for text in STATES.values()]
        native_text=self.tokenizer.decode(candidates[0]).strip()
        if probabilities[0]>.5 and native_text in STATES.values():
            first=self.decoder(prompt)[0,-1].topk(8).indices.tolist()
            for token in first:
                if token in {self.tokenizer.BOS,self.tokenizer.EOS,self.tokenizer.PAD,self.tokenizer.SEP}:continue
                prefix=torch.cat([prompt,torch.tensor([[token]],device=prompt.device)],-1)
                alternative=self.decoder.generate(prefix,max_new_tokens-1,eos)
                candidates.append(alternative[0,prompt.shape[1]:].tolist())
        ranked=[]
        for tokens in {tuple(c) for c in candidates}:
            text=self.tokenizer.decode(list(tokens)).strip()
            if not text or len(tokens)>max_new_tokens:continue
            state=next((key for key,value in STATES.items() if value==text),None)
            prior=probabilities[state] if state is not None else probabilities[0]+probabilities[5]
            target=torch.tensor([tokens],device=prompt.device)
            sequence=torch.cat([prompt,target],-1)
            logits=self.decoder(sequence[:,:-1])[:,prompt.shape[1]-1:]
            likelihood=logits.log_softmax(-1).gather(-1,target.unsqueeze(-1)).sum()
            if getattr(self,'length_normalized',False):likelihood=likelihood/len(tokens)
            value=likelihood+self.threshold*prior.clamp_min(1e-12).log()
            if not torch.isfinite(value):raise ValueError('Escore de fusão não finito.')
            ranked.append((value.item(),tokens))
        if not ranked:raise ValueError('Nenhuma resposta candidata válida.')
        selected=max(ranked,key=lambda pair:pair[0])[1]
        return torch.cat([prompt,torch.tensor([selected],device=prompt.device)],-1)


def evaluation_decoder(decoder,classifier,tokenizer,threshold,*,fusion,length_normalized,factual=None):
    if factual is not None and not fusion:raise ValueError('Decoder factual exige fusão.')
    adapter=FusionDecoder if fusion else RoutedDecoder
    legacy=adapter(decoder,classifier,tokenizer,threshold);legacy.length_normalized=length_normalized
    if factual is None:return legacy
    facts=FusionDecoder(factual,classifier,tokenizer,threshold);facts.length_normalized=length_normalized
    return TypeSelectedDecoder(facts,legacy)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['encoder','head','decoder','data','run-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fusion',action='store_true')
    p.add_argument('--factual-decoder',type=Path)
    p.add_argument('--balanced-states',action='store_true')
    p.add_argument('--paired-presence',action='store_true')
    p.add_argument('--length-normalized',action='store_true')
    p.add_argument('--fresh-head',action='store_true')
    p.add_argument('--label-smoothing',type=float,default=0.)
    p.add_argument('--classifier-from',type=Path)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--patience',type=int,default=6)
    a=p.parse_args()
    if a.factual_decoder and not a.fusion:p.error('Decoder factual exige fusão.')
    if a.run_dir.exists():p.error('Saída deve ser nova.')
    if not 0<=a.label_smoothing<=.5:p.error('Suavização deve estar entre zero e 0,5.')
    if min(a.steps,a.eval_every,a.patience)<=0:p.error('Passos e intervalos devem ser positivos.')
    if a.classifier_from and a.fresh_head:p.error('Escolha classificação preservada ou cabeça nova.')
    torch.set_num_threads(4);torch.manual_seed(42)
    raw=a.encoder.read_bytes();source=torch.load(a.encoder,map_location='cpu',weights_only=True)
    decoder_source=torch.load(a.decoder,map_location='cpu',weights_only=True)
    if source['config']!=decoder_source['config'] or source['merges']!=decoder_source['merges']:p.error('Origens incompatíveis.')
    head=None if a.fresh_head else torch.load(a.head,map_location='cpu',weights_only=True)['head'];classifier=Classifier(source,head).to('mps')
    if a.classifier_from:
        previous=json.loads((a.classifier_from.parent/'run.json').read_text())
        if previous['encoder_sha256']!=hashlib.sha256(raw).hexdigest():p.error('Encoder/tokenizer de origem diferente.')
        state=torch.load(a.classifier_from,map_location='cpu',weights_only=True)
        if state.get('objective')!='experimental_conversation_router' or state.get('promotable') is not False:p.error('Estado experimental incompatível.')
        classifier.load_state_dict(state['classifier'])
    decoder=GPT(GPTConfig(**source['config'])).to('mps');decoder.load_state_dict(decoder_source['model']);decoder.eval().requires_grad_(False)
    factual=None
    if a.factual_decoder:
        factual_source=torch.load(a.factual_decoder,map_location='cpu',weights_only=True)
        if factual_source['config']!=source['config'] or factual_source['merges']!=source['merges']:p.error('Decoder factual incompatível.')
        factual=GPT(GPTConfig(**source['config'])).to('mps');factual.load_state_dict(factual_source['model']);factual.eval().requires_grad_(False)
    tokenizer=ByteBPE(source['merges']);rows,digest=load_instructions(a.data)
    train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if any(not r['task'].startswith('conversation_') for r in rows):p.error('Somente conversa.')
    paired=presence_pools(train) if a.paired_presence else []
    encoded=[];pools={'legacy':{},'context':{}};state_pools=defaultdict(list)
    for i,row in enumerate(train):
        ids,dropped=prepare_prompt(row['history'],row['prompt'],tokenizer,1024,256,'conversation');assert not dropped
        encoded.append(dict(ids=ids));category='context' if row['task'].startswith('conversation_context_answer_') else 'legacy'
        pools[category].setdefault(row['task'],[]).append(i)
        state_pools[semantic_class(row)].append(i)
    if a.balanced_states and set(state_pools)!=set(range(6)):p.error('Todos os seis estados de treino são obrigatórios.')
    optimizer=torch.optim.AdamW([dict(params=classifier.encoder.parameters(),lr=1e-6),dict(params=classifier.head.parameters(),lr=1e-3 if a.fresh_head else 1e-4)],weight_decay=.01)
    generator=torch.Generator().manual_seed(42);a.run_dir.mkdir(parents=True)
    info=dict(method='experimental_lm_class_fusion' if a.fusion else 'experimental_learned_state_router',data=str(a.data),dataset_fingerprint=digest,encoder_sha256=hashlib.sha256(raw).hexdigest(),decoder_sha256=hashlib.sha256(a.decoder.read_bytes()).hexdigest(),head_sha256=hashlib.sha256(a.head.read_bytes()).hexdigest(),test_examples_used=0,promotable=False,note='Classificação aprendida e decoder próprio congelado. Fusão usa probabilidades dos dois modelos e candidatos gerados, sem extrair respostas do texto por regras. Avaliação por respostas completas com histórico gerado. Sem integração na API.')
    info['sampling']='uniform_semantic_class_per_example' if a.balanced_states else 'uniform_legacy_context_family'
    info['paired_presence_groups']=len(paired)
    info['paired_presence_fraction']=.5 if paired else 0.
    info['training_class_counts']={key:len(value) for key,value in state_pools.items()}
    info['length_normalized']=a.length_normalized
    info['head_initialization']='random' if a.fresh_head else 'frozen_source_probe'
    if a.classifier_from:
        info['head_initialization']='continued_experimental_classifier'
        info['classifier_from']=dict(path=str(a.classifier_from),sha256=hashlib.sha256(a.classifier_from.read_bytes()).hexdigest())
    info['optimizer_reset']=True
    info['evaluation_protocol']='learned_state_fusion_v1' if a.fusion else 'learned_state_router_v1'
    if factual is not None:
        info['method']='experimental_type_selected_fusion'
        info['evaluation_protocol']='learned_type_fusion_v2'
        info['factual_decoder']=dict(path=str(a.factual_decoder),sha256=hashlib.sha256(a.factual_decoder.read_bytes()).hexdigest())
    info.update(steps=a.steps,eval_every=a.eval_every,patience=a.patience)
    info['label_smoothing']=a.label_smoothing
    info['parameter_kind']='class_log_probability_weight' if a.fusion else 'confidence_threshold'
    info.update(encoder=str(a.encoder),decoder=str(a.decoder),head=str(a.head))
    (a.run_dir/'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    best=None;stale=0
    for step in range(a.steps+1):
        if step:
            classifier.train()
            if a.balanced_states:
                indices=[state_pools[state][torch.randint(len(state_pools[state]),(),generator=generator).item()] for state in torch.randint(6,(8,),generator=generator).tolist()]
            else:indices=conversation_indices(pools,8,generator)
            if paired:indices[:4]=paired_indices(train,paired,4,generator,label_key='target')
            ids,lengths=unit_batch([encoded[i] for i in indices],'mps',1024)
            labels=torch.tensor([semantic_class(train[i]) for i in indices],device='mps')
            loss=torch.nn.functional.cross_entropy(classifier(ids,lengths),labels,label_smoothing=a.label_smoothing)
            if not torch.isfinite(loss):raise ValueError('Loss não finita.')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(classifier.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%a.eval_every==0 or step==a.steps:
            classifier.eval();results=[]
            for threshold in ([.5,1.,2.] if a.fusion else [.5,.8,.95]):
                predictor=evaluation_decoder(decoder,classifier,tokenizer,threshold,fusion=a.fusion,length_normalized=a.length_normalized,factual=factual)
                with torch.no_grad():result=score(predictor,tokenizer,validation,'mps',1024,256)
                result['criterion']='Respostas completas do pipeline experimental com histórico gerado: classificação aprendida e decoder local; '+('seleção por probabilidades conjuntas, com candidatos do decoder e verbalizadores de estado.' if a.fusion else 'verbalização de estados e decoder greedy nos demais casos.')+' Igualdade exata após strip, orçamento 256. Não é avaliação de decoder greedy isolado ou de qualidade geral.'
                if factual is not None:result['criterion']+=' O tipo aprendido seleciona o decoder factual ou legado.'
                key=quality_key(result);results.append(dict(threshold=threshold,quality=key,result=result))
                print(json.dumps(dict(step=step,threshold=threshold,correct=result['correct'],total=result['total'],quality=key)),flush=True)
            selected=max(results,key=lambda r:r['quality']);(a.run_dir/f'evaluation-{step}.json').write_text(json.dumps(results,ensure_ascii=False))
            if best is None or selected['quality']>best:
                best=selected['quality'];stale=0
                temporary=a.run_dir/'probe.pt.tmp'
                torch.save(dict(classifier={k:v.detach().cpu() for k,v in classifier.state_dict().items()},step=step,objective='experimental_conversation_router',threshold=selected['threshold'],promotable=False),temporary)
                temporary.replace(a.run_dir/'probe.pt')
                (a.run_dir/'evaluation.json').write_text(json.dumps(dict(selected=selected,step=step,promotable=False),ensure_ascii=False,indent=2))
            else:stale+=1
            if best[0] or stale>=a.patience:break
    (a.run_dir/'summary.json').write_text(json.dumps(dict(final_step=step,quality=best,promotable=False),indent=2))


if __name__=='__main__':main()
