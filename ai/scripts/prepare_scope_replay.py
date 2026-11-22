"""Acrescenta consultas pareadas com qualificadores variados, preservando a avaliação."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_query_contrasts import scoped_conversations


def context_key(row):
    return json.dumps([row['history'],row['prompt'].strip().casefold()],ensure_ascii=False)


def merge_scopes(source, additional, namespace):
    reserved={context_key(r) for r in source if r['split']!='train'}
    known={context_key(r):r['target'] for r in source if r['split']=='train'}
    result=list(source);counts=Counter()
    for row in additional:
        if row['task']!='conversation_context_answer_support':continue
        if row['split']!='train' or row['history'] or row['turn_index']!=0:
            raise ValueError('O replay aceita somente consultas isoladas de treino.')
        key=context_key(row)
        if key in reserved:counts['reserved_collision']+=1;continue
        if key in known:
            if known[key]!=row['target']:raise ValueError('Consulta idêntica com alvos conflitantes.')
            counts['duplicate_train']+=1;continue
        known[key]=row['target']
        scene=dict(row['scenario'],paired_presence_group=namespace+'-'+row['scenario']['paired_presence_group'])
        result.append(dict(row,id=namespace+'-'+row['id'],project_id=namespace+'-'+row['project_id'],scenario=scene))
        counts['added_train']+=1
    assert len({r['id'] for r in result})==len(result)
    assert [r for r in result if r['split']!='train']==[r for r in source if r['split']!='train']
    return result,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    source,digest=load_instructions(a.source)
    if any(not r['task'].startswith('conversation_') for r in source):p.error('Base deve conter somente conversas.')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(checkpoint['merges'])
    rows,counts=merge_scopes(source,scoped_conversations(4000),a.output.name)
    assert counts['added_train']>0
    for row in rows[len(source):]:encode_sample(row,tokenizer,checkpoint['config']['block_size'])
    a.output.mkdir(parents=True);target=a.output/'examples.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    load_instructions(target)
    manifest=dict(source_sha256=digest,dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).with_name('prepare_query_contrasts.py').read_bytes()).hexdigest(),
                  generation_groups=4000,counts=dict(counts),splits=dict(Counter(r['split'] for r in rows)),
                  prepositions=['de','por','para','com','em'],validation_unchanged=True,
                  note='Vocabulário e construções de desenvolvimento conhecidos. Consultas completas reservadas excluídas; treino anterior preservado. Novo teste independente após congelamento continua obrigatório.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
