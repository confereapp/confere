"""Matriz editorial de consultas fixas versus parágrafos, sem negativos aleatórios."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from confere_ai.evidence_classifier import examples
from confere_ai.instruct import load_instructions
from confere_ai.qa_source import TRAIN_SHA256
from confere_ai.tokenizer import ByteBPE
from prepare_relation_documents import paragraph_rows, assign_partitions, sha


def matrix_rows(source, annotations):
    docs = {}
    for doc in source['data']: docs.setdefault(doc['title'], []).extend(doc['paragraphs'])
    rows = []; excluded = []
    for doc in annotations['documents']:
        for claim in doc['claims']:
            if (len(claim['labels']) != len(doc['paragraph_indices'])
                    or len(claim['rationales']) != len(claim['labels'])
                    or any(type(v) is not int or v not in {0,1,2} for v in claim['labels'])
                    or any(not r.strip() for r in claim['rationales'])):
                raise ValueError('Matriz de relações incompleta.')
            if claim.get('exclusion_reason'):
                excluded.append(dict(title=doc['title'],query=claim['id'],reason=claim['exclusion_reason']))
                continue
            if len(set(claim['labels'])) < 2: raise ValueError('Consulta sem variação de relação; revisar ou excluir explicitamente.')
            for index,label,rationale in zip(doc['paragraph_indices'],claim['labels'],claim['rationales']):
                text = docs[doc['title']][index]['context']
                item = dict(title=doc['title'],paragraph_index=index,split=doc['split'],anchor=text,
                            claims=[dict(id=claim['id'],text=claim['text'],label=label,rationale=rationale)])
                row = paragraph_rows(source,dict(documents=[item]))[0]
                row['provenance']['relation_matrix_query_id'] = claim['id']
                row['provenance']['relation_matrix_version'] = 2
                rows.append(row)
    return rows, excluded


def query_ceiling(rows):
    groups = defaultdict(Counter)
    for row in rows:
        groups[row['requirement']][json.loads(row['target'])['status']] += 1
    return dict(queries=len(groups),pairs=len(rows),constant_label_queries=sum(len(v)==1 for v in groups.values()),
                max_query_only_correct=sum(max(v.values()) for v in groups.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['qa-source','annotations','replay','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída nova exigida.')
    if sha(a.qa_source)!=TRAIN_SHA256:p.error('Fonte pública alterada.')
    annotations=json.loads(a.annotations.read_text())
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(source['merges'])
    rows,excluded=matrix_rows(json.loads(a.qa_source.read_text()),annotations)
    rows=assign_partitions(rows)
    for row in rows:
        row['provenance']['annotations_sha256']=sha(a.annotations)
        # Falha fechada: perder um par mudaria o balanceamento da consulta inteira.
        examples([row],tokenizer,source['config']['block_size'],readout=source['settings']['readout'])
    for split in ['train','validation','test']:
        if not any(r['split']==split for r in rows):raise ValueError('Split acadêmico vazio.')
    replay,replay_hash=load_instructions(a.replay)
    if any(r['provenance']['kind']!='synthetic_authored' for r in replay):p.error('Replay autoral exigido.')
    projects=defaultdict(list)
    for row in replay:
        if row['split']=='train':projects[row['project_id']].append(row)
    # Projetos inteiros, ordem fixa sem usar previsões; orçamento pequeno de retenção.
    selected=[]
    for key in sorted(projects,key=lambda k:hashlib.sha256(('matrix-replay-v1:'+k).encode()).hexdigest()):
        selected.extend(projects[key])
        if len(selected)>=500:break
    validation=[r for r in replay if r['split']=='validation']
    matrix=list(rows);rows+=selected+validation
    a.output.mkdir(parents=True);path=a.output/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    load_instructions(path)
    manifest=dict(dataset_sha256=sha(path),annotations_sha256=sha(a.annotations),source_sha256=TRAIN_SHA256,
        replay_sha256=replay_hash,replay_train_rows=len(selected),replay_validation_rows=len(validation),
        counts=dict(Counter(r['split']+':'+r['evaluation_profile'] for r in matrix)),
        query_controls={s:query_ceiling([r for r in matrix if r['split']==s]) for s in ['train','validation','test']},
        documents={s:sorted({r['source_document'] for r in matrix if r['split']==s}) for s in ['train','validation','test']},
        exclusions=excluded,random_negatives=False,all_matrix_pairs_fit=True,independent_human_review=False,
        reused_evaluation_is_development=True,public_nli_train_used=False,tccs_used=False,
        note='Matriz editorial revisada localmente. Reserva nova por três títulos; não prova generalização. Fonte e arquitetura próprias preservadas, sem promoção.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
