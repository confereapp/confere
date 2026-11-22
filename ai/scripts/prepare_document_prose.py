"""Contrastes autorais em prosa técnica; nenhum texto dos TCCs entra no corpus."""
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
from prepare_lexical_binding import expected

TOPICS={
    'train':['recuperação de contas','controle de permissões','arquivamento de registros','processamento de pagamentos',
             'importação de tabelas','exportação de cadastros','filtros de pesquisa','histórico de alterações'],
    'validation':['agendamento de consultas','assinatura de arquivos','notificações por mensagem','sincronização de pastas']}
FORMS={
    'train':[('Após a revisão dos componentes, a equipe documentou que a versão entregue oferece {} aos usuários',
              'Após a revisão dos componentes, a equipe documentou que a versão entregue não oferece {} aos usuários'),
             ('A descrição técnica da solução confirma a implementação de {} para as operações do sistema',
              'A descrição técnica da solução confirma que não foi implementado suporte a {} nas operações do sistema'),
             ('No ambiente que foi avaliado, verificou-se a presença de {} na aplicação desenvolvida',
              'No ambiente que foi avaliado, verificou-se a ausência de {} na aplicação desenvolvida')],
    'validation':[('A inspeção do produto entregue constatou que {} está disponível para utilização',
                   'A inspeção do produto entregue constatou que {} não está disponível para utilização')]}
BACKGROUNDS={
    'train':['A discussão sobre qualidade visual orientou a escolha das cores e da disposição dos elementos na interface',
             'A organização do relatório seguiu uma sequência de introdução, contextualização, procedimentos e considerações finais',
             'Os participantes do grupo distribuíram responsabilidades e registraram decisões em reuniões ao longo do semestre',
             'O estudo apresenta uma discussão conceitual da engenharia de software e das atividades realizadas durante o planejamento',
             'INSTITUIÇÃO DE ENSINO\nCURSO SUPERIOR\nTRABALHO DE CONCLUSÃO\nCIDADE UNIVERSITÁRIA\nOrientação acadêmica e apresentação do relatório'],
    'validation':['A monografia foi estruturada em capítulos para organizar a exposição das etapas do trabalho acadêmico',
                  'A apresentação visual dos elementos foi discutida durante as reuniões de acompanhamento da equipe',
                  'BIBLIOGRAFIA\nReferências consultadas no semestre\nOrganização editorial e cidade de publicação']}


def prose_rows(split, count):
    rng=random.Random(171140 if split=='train' else 171141)
    rows=[]
    for i in range(count):
        a,b=rng.sample(TOPICS[split],2);pos,neg=rng.choice(FORMS[split]);mode=i%5
        states=[(a,'atendido')] if mode==0 else [(a,'nao_atendido')] if mode==1 else (
            [(a,'atendido'),(a,'nao_atendido')] if mode==2 else [])
        units=[dict(text=(pos if status=='atendido' else neg).format(topic)+'.',facts=[(topic,status)])
               for topic,status in states]
        if mode==3:
            units.append(dict(text=f'A equipe propôs discutir {a} em uma etapa futura, sem documentar uma decisão ou implementação.',facts=[]))
        if mode!=2:
            units.append(dict(text=pos.format(b)+'.',facts=[(b,'atendido')]))
        units.append(dict(text=rng.choice(BACKGROUNDS[split])+f'\nRegistro editorial {split}-{i}.',facts=[]))
        rng.shuffle(units)
        aliases=[dict(id=f'page:{j+1}',text=u['text']) for j,u in enumerate(units)]
        for j,topic in enumerate([a,b]):
            requirement=(rng.choice(['O documento deve registrar a implementação de {}.',
                                    'O relatório precisa demonstrar a presença de {}.',
                                    'A solução descrita precisa oferecer {}.']).format(topic) if split=='train' else
                         f'O texto deve comprovar que existe suporte a {topic}.')
            target=expected(units,topic);key=f'prose-{split}-{i}-{j}'
            row=dict(id=key,project_id=f'prose-{split}-{i}',split=split,task='grounded_'+target['status'],
                     grounding_schema=2,evaluation_profile=f'prose_{mode}',requirement=requirement,
                     units=copy.deepcopy(aliases),document=key+'.txt',prompt=prompt_for(requirement,aliases,2),
                     target=json.dumps(target,ensure_ascii=False),scenario=dict(units=copy.deepcopy(units),feature=topic),
                     provenance=dict(kind='synthetic_authored',note='Prosa e estados construídos localmente; sem texto dos TCCs ou revisão humana independente.'))
            assert judge(row['target'],row)['exact_match']
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    base,base_hash=load_instructions(a.base);rng=random.Random(42)
    train=[r for r in base if r['split']=='train'];validation=[r for r in base if r['split']=='validation']
    replay=rng.sample(train,min(len(train),5000))
    rows=replay+validation+prose_rows('train',1500)+prose_rows('validation',80)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(checkpoint['merges'])
    examples(rows,tokenizer,checkpoint['config']['block_size'],readout=checkpoint['settings']['readout'])
    a.output.mkdir(parents=True);path=a.output/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));load_instructions(path)
    manifest=dict(base_sha256=base_hash,dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  counts=dict(Counter(r['split'] for r in rows)),replay_rows=len(replay),real_documents_used=False,
                  external_weights_used=False,original_validation_unchanged=True,
                  note='Títulos, redações e tópicos reservados na validação nova. Supervisão autoral explícita por unidade; não rotula automaticamente parágrafos reais como negativos. Desenvolvimento, não teste cego.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
