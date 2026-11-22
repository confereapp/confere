"""Amplia as redações dos critérios somente no treino; alvos e avaliação intactos."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import torch
from confere_ai.evidence_classifier import examples
from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE

TEMPLATES={
    'presence': ['O registro deve confirmar a presença de {}.',
                 'O relatório deve registrar que {} faz parte da versão atual.',
                 'É necessário que a versão atual contenha {}.',
                 'A entrega descrita deve incluir {}.',
                 'O texto precisa indicar que {} está presente no produto.',
                 'A documentação precisa registrar {} como componente presente.'],
    'absence': ['O registro deve confirmar a ausência de {}.',
                'O documento deve registrar que {} está ausente da versão atual.',
                'O relatório precisa declarar que a versão atual está sem {}.',
                'Deve ficar documentado que {} não faz parte da entrega.',
                'O produto descrito não deve conter {}.',
                'O registro deve indicar que a versão atual não inclui {}.'],
    'mention': ['O registro deve mencionar {}.',
                'Deve haver uma menção a {} na documentação.',
                'O relatório deve mencionar {}.',
                'É necessário citar {} no relatório.',
                'O texto deve fazer referência a {}.',
                'A documentação precisa conter uma menção a {}.']}


def paraphrase(row,rng):
    if row['split']!='train' or row.get('scenario',{}).get('criterion') not in TEMPLATES:return row
    result=copy.deepcopy(row);scenario=row['scenario']
    requirement=rng.choice(TEMPLATES[scenario['criterion']]).format(scenario['topic'])
    result.update(requirement=requirement,prompt=prompt_for(requirement,result['units'],2))
    result['provenance']['note']+=' Redação da consulta ampliada; texto-fonte e alvo intactos.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    before,base_hash=load_instructions(a.base);rng=random.Random(171160)
    rows=[paraphrase(r,rng) for r in before]
    assert [r for r in rows if r['split']!='train']==[r for r in before if r['split']!='train']
    assert all(a['target']==b['target'] and a['units']==b['units'] for a,b in zip(before,rows))
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    examples(rows,ByteBPE(checkpoint['merges']),checkpoint['config']['block_size'],readout=checkpoint['settings']['readout'])
    a.output.mkdir(parents=True);path=a.output/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));load_instructions(path)
    manifest=dict(base_sha256=base_hash,dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  changed_train_queries=sum(x['prompt']!=y['prompt'] for x,y in zip(before,rows)),
                  validation_unchanged=True,targets_unchanged=True,source_texts_unchanged=True,
                  real_documents_used=False,external_weights_used=False,label_scope='component_inclusion_v2',
                  note='Paráfrases autorais para expor os operadores semânticos no treino; avaliação já conhecida continua desenvolvimento.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
