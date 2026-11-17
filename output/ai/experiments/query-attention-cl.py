import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'ai/scripts')
from confere_ai.evidence_classifier import EvidencePredictor,examples,unit_batch,paired_indices
from confere_ai.instruct import load_instructions,acceptance
from confere_ai.generative_selection import quality_key
from train_evidence_classifier import score

class AttentionReadout(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base.eval().requires_grad_(False)
        width=base.backbone.config.width
        self.attention=nn.MultiheadAttention(width,base.backbone.config.heads,batch_first=True,dropout=0.)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
    @property
    def backbone(self):return self.base.backbone
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def both(self,ids,lengths):
        with torch.no_grad():
            hidden=self.base.backbone(ids);last=hidden[torch.arange(len(ids),device=ids.device),lengths-1]
            teacher=self.base.head(last)
        padding=torch.arange(ids.shape[1],device=ids.device)[None,:]>=lengths[:,None]
        extra,_=self.attention(last[:,None,:],hidden,hidden,key_padding_mask=padding,need_weights=False)
        return self.base.head(last+extra[:,0]),teacher
    def forward(self,ids,lengths):return self.both(ids,lengths)[0]
    def evidence_scores(self,logits):return self.base.evidence_scores(logits)

torch.set_num_threads(4);torch.manual_seed(42)
root=Path('ai/runs/evidence-attention-cl-2026-11-17');assert not root.exists()
source=Path('ai/runs/evidence-calibrated-ci-2026-11-17/best.pt');payload=source.read_bytes()
checkpoint=torch.load(source,map_location='cpu',weights_only=True)
assert checkpoint['settings'].get('state_mode','independent_binary')=='independent_binary'
predictor=EvidencePredictor(checkpoint,'mps');model=AttentionReadout(predictor.model).to('mps')
data=Path('ai/data/private/document-control-cc-2026-11-17/examples.jsonl');rows,digest=load_instructions(data)
train=examples([r for r in rows if r['split']=='train'],predictor.tokenizer,predictor.config.block_size,readout=predictor.readout)
validation=[r for r in rows if r['split']=='validation'];assert len(validation)==410
groups=defaultdict(list)
for i,item in enumerate(train):groups[item['source_key']].append(i)
pools=list(groups.values());generator=torch.Generator().manual_seed(42)
ids,lengths=unit_batch(train[:8],'mps',predictor.config.block_size)
with torch.no_grad():student,teacher=model.both(ids,lengths)
torch.testing.assert_close(student,teacher,rtol=0,atol=0)
root.mkdir();(root/'source.pt').write_bytes(payload)
info=dict(data=str(data),dataset_fingerprint=digest,source=str(source),source_sha256=hashlib.sha256(payload).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),objective='experimental_attention_readout',promotable=False,
    frozen_encoder_and_base_head=True,steps=6000,eval_every=500,batch_size=8,learning_rate=.0001,teacher_weight=1.,test_examples_used=0,
    note='Residual query attention starts as exact parent. Only attention trained. Teacher preservation on correctly classified training bits; no defaults or inference loader changed.')
(root/'run.json').write_text(json.dumps(info,indent=2))
predictor.model=model;optimizer=torch.optim.AdamW(model.attention.parameters(),lr=.0001,weight_decay=.01)
thresholds=torch.tensor(predictor.thresholds,device='mps');best=None
for step in range(6001):
    if step:
        model.train();indices=paired_indices(train,pools,8,generator)
        batch=[train[i] for i in indices];ids,lengths=unit_batch(batch,'mps',predictor.config.block_size)
        labels=torch.tensor([r['labels'] for r in batch],device='mps')
        student,teacher=model.both(ids,lengths);prob=teacher.sigmoid().detach()
        correct=(prob>=thresholds)==labels.bool()
        loss=F.binary_cross_entropy_with_logits(student,labels)
        if correct.any():loss=loss+F.binary_cross_entropy_with_logits(student[correct],prob[correct])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        torch.nn.utils.clip_grad_norm_(model.attention.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    if step%500==0:
        model.eval();result=score(model,[],validation,'mps',predict_units=predictor.predict_units);key=quality_key(result)
        (root/f'evaluation-{step}.json').write_text(json.dumps(dict(quality=key,result=result),ensure_ascii=False))
        if best is None or key>best:
            best=key
            temporary=root/'probe.pt.tmp';torch.save(dict(attention=model.attention.state_dict(),step=step,objective=info['objective'],source_sha256=info['source_sha256'],promotable=False),temporary);temporary.replace(root/'probe.pt')
            (root/'evaluation.json').write_text(json.dumps(dict(selected=result,step=step,quality=key,acceptance=acceptance(result),promotable=False),ensure_ascii=False))
        print(json.dumps(dict(step=step,correct=result['correct'],quality=key)),flush=True)
        if result['correct']==result['total']:break
(root/'summary.json').write_text(json.dumps(dict(final_step=step,quality=best,promotable=False),indent=2))
