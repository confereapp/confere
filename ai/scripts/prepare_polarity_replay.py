"""Contrastes de disponibilidade por frase; avaliações anteriores permanecem separadas."""
import argparse
import json
from pathlib import Path
from confere_ai.instruct import load_instructions
from prepare_query_contrasts import documents

FORMS=[
    ('{} está disponível no projeto.','{} está indisponível no projeto.'),
    ('A equipe disponibiliza {}.','A equipe não disponibiliza {}.'),
    ('O sistema dispõe de {}.','O sistema não dispõe de {}.'),
    ('O projeto conta com {}.','O projeto não conta com {}.'),
    ('Já existe {}.','Ainda não existe {}.'),
    ('Constatamos a implementação de {}.','Constatamos a falta de implementação de {}.'),
    ('Foi entregue um componente funcional para {}.','Ainda falta desenvolver o componente de {}.'),
    ('O recurso {} está implementado e disponível.','O recurso {} não está implementado nem entregue.'),
    ('O sistema oferece suporte funcional para {}.','O sistema não oferece suporte funcional para {}.'),
    ('A solução oferece {} para os usuários.','A solução não oferece {} para os usuários.'),
    ('A versão atual permite utilizar {}.','A versão atual não permite utilizar {}.'),
    ('A implementação entregue assegura {}.','A implementação entregue está sem {}.'),
    ('O módulo de {} está funcionando.','O módulo de {} não foi implementado.'),
    ('O software fornece {}.','O software não fornece {}.'),
    ('Encontramos a funcionalidade de {} no produto.','Não encontramos a funcionalidade de {} no produto.'),
    ('O relatório confirma que {} está disponível.','O relatório confirma que {} está indisponível.'),
    ('O documento informa que {} está implementado.','O documento informa que {} não está implementado.'),
    ('A revisão registra que {} está disponível.','A revisão registra que {} está indisponível.'),
    ('Foi verificado que existe {}.','Foi verificado que não existe {}.'),
    ('Consta na documentação que {} foi entregue.','Consta na documentação que {} não foi entregue.'),
    ('A equipe comprovou que {} funciona.','A equipe comprovou que {} não funciona.'),
    ('A comissão atesta o funcionamento de {}.','A comissão atesta a ausência de {}.'),
    ('A documentação atesta que existe {}.','A documentação atesta que não existe {}.'),
    ('O texto atesta que {} funciona.','O texto atesta que {} não funciona.'),
    ('O relatório atesta a disponibilidade de {}.','O relatório atesta a indisponibilidade de {}.'),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','topics-manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída já existe.')
    rows,digest=load_instructions(a.source)
    if any(not r['task'].startswith('grounded_') for r in rows):p.error('Fonte deve conter apenas revisão documental.')
    topics=json.loads(a.topics_manifest.read_text())['topics']
    if not isinstance(topics,list) or any(not isinstance(t,str) or not t.strip() for t in topics) or len(set(topics))<2:
        p.error('Recursos distintos e não vazios são obrigatórios.')
    added=documents(8000,topics=topics,forms=FORMS,sentence_units=True)
    for row in added:row.update(id=a.output.name+'-'+row['id'],project_id=a.output.name+'-'+row['project_id'])
    seen={r['prompt'].strip().casefold():r for r in rows};retained=[];skipped=0
    for row in added:
        key=row['prompt'].strip().casefold();previous=seen.get(key)
        if previous is not None:
            if previous['split']!='train' or json.loads(previous['target'])!=json.loads(row['target']):
                raise ValueError('Contexto repetido entre splits ou com alvos diferentes.')
            skipped+=1;continue
        seen[key]=row;retained.append(row)
    result=rows+retained
    assert len({r['id'] for r in result})==len(result)
    assert [r for r in result if r['split']!='train']==[r for r in rows if r['split']!='train']
    a.output.mkdir(parents=True);f=a.output/'examples.jsonl'
    f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    _,fingerprint=load_instructions(f)
    manifest=dict(source_sha256=digest,dataset_sha256=fingerprint,added_train=len(retained),duplicate_train_skipped=skipped,
                  train=sum(r['split']=='train' for r in result),validation=sum(r['split']=='validation' for r in result),
                  forms=FORMS,topics=topics,note='Polaridade lexical por frase e recurso consultado. Diagnósticos reprovados continuam desenvolvimento; novo teste independente obrigatório.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['forms','topics']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
