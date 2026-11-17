import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'ai/scripts')
from prepare_api_batches import local_rows
from calibrate_evidence import evaluate_thresholds
from confere_ai.tokenizer import ByteBPE
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import acceptance

root=Path('ai/runs/evidence-regions-ck-verified-2026-11-17');assert not root.exists()
audit=json.loads(Path('output/ai/ci-probability-margins-2026-11-17.json').read_text())
source=Path('ai/runs/evidence-calibrated-ci-2026-11-17/best.pt');payload=source.read_bytes()
assert hashlib.sha256(payload).hexdigest()==audit['source_sha256']
checkpoint=torch.load(source,map_location='cpu',weights_only=True)
data=Path('ai/data/private/document-control-cc-2026-11-17/examples.jsonl')
assert hashlib.sha256(data.read_bytes()).hexdigest()==audit['dataset_sha256']
rows=[json.loads(line) for line in data.open() if '"split": "validation"' in line]
assert len(rows)==410
by_id={r['id']:i for i,r in enumerate(rows)};by_prompt={r['prompt']:i for i,r in enumerate(rows)}
probabilities=[s['probabilities'] for s in audit['scores']];labels=np.array([s['labels'] for s in audit['scores']],dtype=bool)
values=np.array(probabilities);row_units=[[] for _ in rows]
for j,s in enumerate(audit['scores']):row_units[by_id[s['row_id']]].append(j)
assert all(row_units)
def candidates(v,original):
    edges=np.unique(np.concatenate([[0.],v,[1.]]))
    return np.unique(np.concatenate([(edges[:-1]+edges[1:])/2,[original]]))
ps=candidates(values[:,0],.3);ns=candidates(values[:,1],.5)
def view(thresholds,axis):
    pred=values[None,:,axis]>=thresholds[:,None]
    correct=np.stack([(pred[:,idx]==labels[None,idx,axis]).all(1) for idx in row_units],axis=1)
    any_value=np.stack([pred[:,idx].any(1) for idx in row_units],axis=1)
    return correct,any_value
pc,pa=view(ps,0);nc,na=view(ns,1)
shape=(len(ps),len(ns));local_pass=np.ones(shape,dtype=bool)
profiles={r['evaluation_profile'] for r in rows}
for profile in profiles:
    idx=[i for i,r in enumerate(rows) if r['evaluation_profile']==profile]
    count=pc[:,idx].astype(np.int32)@nc[:,idx].astype(np.int32).T
    local_pass&=count>=np.ceil(.9*len(idx))
nonpositive=[i for i,r in enumerate(rows) if json.loads(r['target'])['status']!='atendido']
fp=pa[:,nonpositive].astype(np.int32)@(~na[:,nonpositive]).astype(np.int32).T
local_pass&=fp==0
full_pass=np.ones(shape,dtype=bool);tokenizer=ByteBPE(checkpoint['merges']);baseline_counts={}
pi=int(np.where(ps==.3)[0][0]);ni=int(np.where(ns==.5)[0][0])
for name,path in [('original','ai/data/private/api-batches-2026-11-11/validation-projects.json'),('previous-extra','ai/data/private/evidence-project-heldout-2026-11-16/cases.json'),('new-development','ai/data/private/evidence-independent-al-2026-11-16/cases.json')]:
    correct=0;total=0
    actual=json.loads(Path(f'ai/runs/evidence-calibrated-ci-2026-11-17/full-{name}.json').read_text())
    actual_by_id={r['id']:r for r in actual['results']}
    for scene in json.loads(Path(path).read_text()):
        local=local_rows(scene,tokenizer,checkpoint['config']['block_size']);assert local
        for requirement_index,(requirement,gold) in enumerate(zip(scene['requirements'],scene['expected'])):
            text=requirement['requirement'] if isinstance(requirement,dict) else requirement
            idx=[by_prompt[r['prompt']] for r in local if r['requirement']==text];assert idx
            positive=pa[:,idx].any(1)[:,None];negative=na[:,idx].any(1)[None,:]
            if gold['status']=='atendido':valid=positive&~negative
            elif gold['status']=='nao_atendido':valid=~positive&negative
            elif gold['conflicting_evidence']:valid=positive&negative
            else:valid=~positive&~negative
            assert bool(valid[pi,ni])==bool(actual_by_id[scene['id']]['requirements'][requirement_index]['correct'])
            full_pass&=valid;correct+=int(valid[pi,ni]);total+=1
    actual=json.loads(Path(f'ai/runs/evidence-calibrated-ci-2026-11-17/full-{name}.json').read_text())
    assert (correct,total)==(actual['correct_requirements'],actual['total_requirements'])
    baseline_counts[name]=[correct,total]
feasible=np.argwhere(local_pass&full_pass)
root.mkdir();info=dict(data=str(data),dataset_fingerprint=audit['dataset_sha256'],source=str(source),source_sha256=audit['source_sha256'],
    positive_regions=len(ps),negative_regions=len(ns),feasible_regions=len(feasible),baseline_full=baseline_counts,
    baseline_verified_per_requirement=True,training_steps=0,test_examples_used=0,adopted=False,note='Two global thresholds only. Exhaustive score-induced development regions, including known complete documents. Must verify through actual pipeline and fresh independent cohort.')
(root/'run.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
if len(feasible):
    total=pc.astype(np.int32)@nc.astype(np.int32).T
    best=max(feasible.tolist(),key=lambda xy:(total[tuple(xy)],-abs(ps[xy[0]]-.3)-abs(ns[xy[1]]-.5)))
    thresholds=[float(ps[best[0]]),float(ns[best[1]])]
    result=evaluate_thresholds(rows,probabilities,thresholds);assert acceptance(result)['passed']
    checkpoint['settings']=dict(checkpoint['settings'],thresholds=thresholds,calibration='development_exact_regions_v1')
    (root/'source.pt').write_bytes(payload);torch.save(checkpoint,root/'best.pt')
    (root/'evaluation.json').write_text(json.dumps(dict(selected=result,quality=quality_key(result),acceptance=acceptance(result),thresholds=thresholds,adopted=False),ensure_ascii=False,indent=2))
    print(json.dumps(dict(thresholds=thresholds,correct=result['correct'],quality=quality_key(result))),flush=True)
