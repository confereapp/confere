"""Sonda de relação consulta/trecho com encoder próprio congelado; não promovível."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from confere_ai.evidence_classifier import EvidencePredictor, encode_unit, examples, unit_batch, sentence_parts, response, paired_indices, unit_weighted_query_pools
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import load_instructions
from confere_ai.project_review import review_extracted
from train_evidence_classifier import score

OBJECTIVE='experimental_query_conditioned_evidence_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenPairs:
    def __init__(self,source,device):
        self.base=EvidencePredictor(source,device)
        if self.base.model.state_mode!='independent_binary':raise ValueError('A sonda exige duas decisões binárias na origem.')
        for p in self.base.model.parameters():p.requires_grad_(False)
        self.embedding=self.base.model.backbone.token_embedding.weight.detach().cpu().clone()
        # shortcut: caches vivem por ensaio; limitar antes de integrar a serviço persistente.
        self.cache={};self.query_cache={};self.mean_cache={}

    def mean(self,text):
        if text not in self.mean_cache:
            ids=self.base.tokenizer.encode(text)
            self.mean_cache[text]=(self.embedding[ids].mean(0) if ids else torch.zeros(self.base.config.width))
        return self.mean_cache[text]

    @torch.no_grad()
    def features(self,pairs):
        if any(not isinstance(q,str) or not q.strip() for q,t in pairs):raise ValueError('Consulta não vazia exigida.')
        missing=list(dict.fromkeys(pair for pair in pairs if pair not in self.cache))
        queries=list(dict.fromkeys(q for q,t in missing if q not in self.query_cache))
        for offset in range(0,len(queries),8):
            group=queries[offset:offset+8]
            items=[dict(ids=[self.base.tokenizer.BOS]+self.base.tokenizer.encode(q)+[self.base.tokenizer.SEP]) for q in group]
            if any(len(i['ids'])>self.base.config.block_size for i in items):raise ValueError('Consulta excede contexto.')
            ids,lengths=unit_batch(items,self.base.device,self.base.config.block_size)
            hidden=self.base.model.features(ids,lengths).cpu()
            for q,h in zip(group,hidden):self.query_cache[q]=h
        for offset in range(0,len(missing),8):
            group=missing[offset:offset+8]
            items=[dict(ids=encode_unit(q,t,self.base.tokenizer,self.base.config.block_size,readout=self.base.readout)) for q,t in group]
            ids,lengths=unit_batch(items,self.base.device,self.base.config.block_size)
            hidden=self.base.model.features(ids,lengths)
            logits=self.base.model.head(hidden).cpu();hidden=hidden.cpu()
            for (q,t),h,b in zip(group,hidden,logits):
                qmean,tmean=self.mean(q),self.mean(t)
                x=torch.cat([qmean,tmean,(qmean-tmean).abs(),qmean*tmean,self.query_cache[q],h,b])
                self.cache[q,t]=(x,b)
            if len(missing)>1000 and offset%1024==0:
                print(json.dumps(dict(phase='frozen_features',complete=offset+len(group),total=len(missing))),flush=True)
        return torch.stack([self.cache[p][0] for p in pairs]),torch.stack([self.cache[p][1] for p in pairs])


class ConditionalHead(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        if mean.ndim!=1 or mean.shape!=scale.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Normalização inválida.')
        self.register_buffer('mean',mean.detach().float())
        self.register_buffer('scale',scale.detach().float())
        self.net=nn.Sequential(nn.Linear(len(mean),128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base_logits):
        return base_logits+self.net((x-self.mean)/self.scale)


class ConditionedPredictor:
    def __init__(self,encoder,head):
        self.encoder=encoder;self.head=head

    @torch.no_grad()
    def probabilities(self,query,units):
        if not units or len(units)>4 or len({u['id'] for u in units})!=len(units):raise ValueError('Uma a quatro unidades distintas são exigidas.')
        parts=[sentence_parts(u['text']) if self.encoder.base.segmentation=='sentences_v1' else [u['text']] for u in units]
        x,b=self.encoder.features([(query,t) for group in parts for t in group])
        values=self.head(x,b).sigmoid()
        if not torch.isfinite(values).all():raise ValueError('Probabilidades não finitas.')
        result=[];offset=0
        for group in parts:
            result.append(values[offset:offset+len(group)].amax(0).tolist());offset+=len(group)
        return result

    def predict_units(self,query,units):
        return response(units,self.probabilities(query,units),self.encoder.base.thresholds)


def load_probe(path,device='mps'):
    path=Path(path);state=torch.load(path,map_location='cpu',weights_only=True)
    if state['settings'].get('objective')!=OBJECTIVE:raise ValueError('Protocolo experimental não suportado.')
    source_path=path.parent/'source.pt'
    if sha(source_path)!=state['source_sha256']:raise ValueError('Origem congelada foi alterada.')
    encoder=FrozenPairs(torch.load(source_path,map_location='cpu',weights_only=True),device)
    dimension=6*encoder.base.config.width+2
    head=ConditionalHead(torch.zeros(dimension),torch.ones(dimension))
    head.load_state_dict(state['head']);head.eval()
    return ConditionedPredictor(encoder,head)


def review_conditioned(documents,requirements,path,device='mps'):
    torch.set_num_threads(4);path=Path(path);before=sha(path);predictor=load_probe(path,device)
    base=predictor.encoder.base
    result=review_extracted(documents,requirements,base.tokenizer,base.config.block_size,None,2,256,
                            predict_units=predictor.predict_units)
    assert sha(path)==before
    result['model']=dict(checkpoint=str(path),sha256=before,inference_kind=OBJECTIVE,
                         source_sha256=sha(path.parent/'source.pt'),device=device,
                         context_size=base.config.block_size,promotable=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args()
    if a.run_dir.exists() or a.steps<=0:p.error('Saída nova e passos positivos exigidos.')
    torch.set_num_threads(4);torch.manual_seed(42)
    rows,digest=load_instructions(a.data)
    if any(not r['task'].startswith('grounded_') for r in rows):p.error('Somente dados documentais.')
    source=torch.load(a.source,map_location='cpu',weights_only=True)
    encoder=FrozenPairs(source,a.device)
    train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    lookup={(r['id'],u['id']):(r['requirement'],u['text']) for r in train for u in r['units']}
    items=examples(train,encoder.base.tokenizer,encoder.base.config.block_size,readout=encoder.base.readout)
    x,b=encoder.features([lookup[i['row_id'],i['unit']] for i in items]);labels=torch.tensor([i['labels'] for i in items])
    head=ConditionalHead(x.mean(0),x.std(0,unbiased=False).clamp_min(.001))
    predictor=ConditionedPredictor(encoder,head)
    pairs=[(r['requirement'],part) for r in validation for u in r['units']
           for part in (sentence_parts(u['text']) if encoder.base.segmentation=='sentences_v1' else [u['text']])]
    encoder.features(pairs)
    native=score(encoder.base.model,[],validation,a.device,encoder.base.thresholds,predict_units=encoder.base.predict_units)
    baseline=score(head,[],validation,'cpu',encoder.base.thresholds,predict_units=predictor.predict_units)
    assert [(r['id'],r['actual']) for r in baseline['samples']]==[(r['id'],r['actual']) for r in native['samples']]
    a.run_dir.mkdir(parents=True)
    frozen={k:source[k] for k in ('config','merges','classifier','settings','step')}
    torch.save(frozen,a.run_dir/'source.pt');frozen_hash=sha(a.run_dir/'source.pt')
    (a.run_dir/'run.json').write_text(json.dumps(dict(data=str(a.data),dataset_fingerprint=digest,source=str(a.source),
        original_source_sha256=sha(a.source),source_sha256=frozen_hash,objective=OBJECTIVE,encoder_frozen=True,
        train_units=len(items),validation_rows=len(validation),normalization_from_train_only=True,
        baseline_equivalence_verified=True,head_device='cpu',encoder_device=a.device,
        sampling='paired_queries_uniform_unit_then_opposite_label',steps=a.steps,script_sha256=sha(Path(__file__)),
        promotable=False,note='Sonda residual treinada; sem interpretação por palavras-chave, sem pesos externos ou integração na API.'),indent=2))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(42);pools=unit_weighted_query_pools(items)
    teacher=b.sigmoid();good=((teacher>=torch.tensor(encoder.base.thresholds))==labels.bool()).float()
    best=None;selected=None;selected_step=None
    for step in range(a.steps+1):
        if step:
            head.train();indices=paired_indices(items,pools,128,generator)
            logits=head(x[indices],b[indices])
            anchor=(F.binary_cross_entropy_with_logits(logits,teacher[indices],reduction='none')*good[indices]).sum()/good[indices].sum().clamp_min(1)
            loss=F.binary_cross_entropy_with_logits(logits,labels[indices])+anchor
            if not torch.isfinite(loss):raise ValueError('Loss não finita.')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%250==0 or step==a.steps:
            result=baseline if step==0 else score(head,[],validation,'cpu',encoder.base.thresholds,predict_units=predictor.predict_units)
            key=quality_key(result)
            (a.run_dir/f'evaluation-{step}.json').write_text(json.dumps(dict(quality=key,result=result),ensure_ascii=False))
            state=dict(settings=dict(objective=OBJECTIVE,promotable=False),head=head.state_dict(),
                       source_sha256=frozen_hash,dataset_fingerprint=digest,step=step)
            temporary=a.run_dir/'last.pt.tmp';torch.save(state,temporary);temporary.replace(a.run_dir/'last.pt')
            if best is None or key>best:
                best,selected,selected_step=key,result,step
                temporary=a.run_dir/'probe.pt.tmp';torch.save(state,temporary);temporary.replace(a.run_dir/'probe.pt')
            print(json.dumps(dict(step=step,correct=result['correct'],total=result['total'],quality=key)),flush=True)
            if result['correct']==result['total']:break
    assert all(p.grad is None for p in encoder.base.model.parameters())
    (a.run_dir/'evaluation.json').write_text(json.dumps(dict(selected_step=selected_step,quality=best,
        baseline=baseline,selected=selected,adopted=False),ensure_ascii=False))


if __name__=='__main__':main()
