"""Distingue estado do recurso e critério documental usando o mesmo texto."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import torch
from confere_ai.evidence_classifier import examples
from confere_ai.grounding import prompt_for, judge
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from prepare_document_prose import TOPICS, FORMS, BACKGROUNDS


def criterion_rows(split,count):
    rng=random.Random(171150 if split=='train' else 171151);rows=[]
    for i in range(count):
        a,b=rng.sample(TOPICS[split],2);pos,neg=rng.choice(FORMS[split]);mode=i%5
        facts=[(a,'atendido')] if mode==0 else [(a,'nao_atendido')] if mode==1 else (
            [(a,'atendido'),(a,'nao_atendido')] if mode==2 else [])
        parts=[dict(text=(pos if state=='atendido' else neg).format(topic)+'.',topic=topic,state=state)
               for topic,state in facts]
        if mode==3:parts.append(dict(text=f'O tema {a} foi mencionado no planejamento, sem decisão sobre sua implementação.',topic=a,state=None))
        parts.append(dict(text=pos.format(b)+'.',topic=b,state='atendido'))
        parts.append(dict(text=rng.choice(BACKGROUNDS[split])+f'\nControle de critérios {split}-{i}.',topic=None,state=None))
        rng.shuffle(parts);units=[dict(id=f'page:{j+1}',text=u['text']) for j,u in enumerate(parts)]
        templates=({'presence':'O registro deve confirmar a presença de {}.',
                    'absence':'O registro deve confirmar a ausência de {}.',
                    'mention':'O registro deve mencionar {}.'} if split=='train' else
                   {'presence':'O documento precisa informar que {} está disponível.',
                    'absence':'O documento precisa informar que {} está ausente.',
                    'mention':'O documento precisa conter uma menção a {}.'})
        for j,topic in enumerate([a,b]):
            for kind,template in templates.items():
                evidence=[]
                for k,u in enumerate(parts,1):
                    if u['topic']!=topic:continue
                    state=('atendido' if kind=='mention' else u['state'] if kind=='presence' else
                           'nao_atendido' if u['state']=='atendido' else 'atendido' if u['state']=='nao_atendido' else None)
                    if state:evidence.append(dict(status=state,unit=f'page:{k}'))
                statuses={e['status'] for e in evidence}
                status=next(iter(statuses)) if len(statuses)==1 else 'evidencia_insuficiente'
                key=f'criterion-{split}-{i}-{j}-{kind}';requirement=template.format(topic)
                row=dict(id=key,project_id=f'criterion-{split}-{i}',split=split,task='grounded_'+status,
                    grounding_schema=2,evaluation_profile='criterion_'+kind,requirement=requirement,
                    document=key+'.txt',units=copy.deepcopy(units),prompt=prompt_for(requirement,units,2),
                    target=json.dumps(dict(status=status,evidence=evidence),ensure_ascii=False),
                    scenario=dict(criterion=kind,topic=topic,parts=copy.deepcopy(parts)),
                    provenance=dict(kind='synthetic_authored',note='Critério explícito com alvo derivado por construção; mencionar ausência não equivale a exigir presença. Sem texto dos TCCs.'))
                assert judge(row['target'],row)['exact_match'];rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    base,base_hash=load_instructions(a.base)
    train=[r for r in base if r['split']=='train'];validation=[r for r in base if r['split']=='validation']
    rows=random.Random(42).sample(train,min(4000,len(train)))+validation+criterion_rows('train',1000)+criterion_rows('validation',30)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    examples(rows,ByteBPE(checkpoint['merges']),checkpoint['config']['block_size'],readout=checkpoint['settings']['readout'])
    a.output.mkdir(parents=True);path=a.output/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));load_instructions(path)
    manifest=dict(base_sha256=base_hash,dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  counts=dict(Counter(r['split'] for r in rows)),original_validation_unchanged=True,
                  real_documents_used=False,external_weights_used=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Contrastes de critério documental. Validação conhecida de desenvolvimento, não teste cego. Nenhum TCC usado como treino.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
