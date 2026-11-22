"""Piloto FaQuAD: somente treino oficial, reserva por título, sem cortar contexto."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from confere_ai.instruct import load_instructions
from confere_ai.qa_source import SOURCE, REVISION, TRAIN_SHA256, validate_provenance
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample


def public_rows(data):
    titles = sorted({a['title'] for a in data['data']}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    reserved = set(titles[:max(1, len(titles)//5)])
    reserved_contexts = {p['context'] for a in data['data'] if a['title'] in reserved for p in a['paragraphs']}
    for article in data['data']:
        split = 'validation' if article['title'] in reserved else 'train'
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if split == 'train' and context in reserved_contexts:
                continue
            for qa in paragraph['qas']:
                # shortcut: exact match usa o primeiro alvo; avaliar variantes antes de alegar qualidade geral.
                answer = qa['answers'][0]
                row = dict(id='faquad-'+qa['id'], project_id='faquad-'+qa['id'], split=split,
                           task='public_dialogue' if split == 'train' else 'conversation_context_answer_faquad',
                           history=[], turn_index=0, response_tokens=256,
                           prompt=f"Contexto: {context}\nPergunta: {qa['question']}\nResponda usando apenas o texto.",
                           target=answer['text'], provenance=dict(kind='public_extractive_qa', source=SOURCE,
                           revision=REVISION, train_sha256=TRAIN_SHA256, license='CC-BY-4.0',
                           title=article['title'], context=context, question=qa['question'],
                           answer_start=answer['answer_start'], source_row_id=qa['id']))
                yield row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','base','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Saída deve ser nova.')
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TRAIN_SHA256: p.error('Snapshot inesperado.')
    base,base_hash=load_instructions(a.base)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(checkpoint['merges']);counts=Counter();rows=[]
    pools=defaultdict(list)
    for row in base:
        if row['split']=='train': pools[row['task']].append(row)
        else: rows.append(row)
    rng=random.Random(42)
    for task,pool in sorted(pools.items()):
        for row in rng.sample(pool,min(300,len(pool))):
            # Cada replay conserva seu histórico fornecido, sem fingir uma cadeia amostrada completa.
            rows.append(dict(row,project_id='replay-'+row['id'],turn_index=0) if 'history' in row else row)
    seen={(json.dumps(r.get('history',[])),r['prompt'].strip().casefold()) for r in rows}
    for row in public_rows(json.loads(raw)):
        try: validate_provenance(row)
        except ValueError: counts['unaligned']+=1;continue
        try: encode_sample(row,tokenizer,checkpoint['config']['block_size'])
        except ValueError: counts['too_long']+=1;continue
        key=(json.dumps(row['history']),row['prompt'].strip().casefold())
        if key in seen: counts['duplicate']+=1;continue
        seen.add(key); rows.append(row);counts['public_'+row['split']]+=1
    assert counts['public_train'] and counts['public_validation']
    public=[r for r in rows if r['provenance']['kind']=='public_extractive_qa']
    for field in ('title','context'):
        assert not ({r['provenance'][field] for r in public if r['split']=='train'} &
                    {r['provenance'][field] for r in public if r['split']!='train'})
    a.output.mkdir(parents=True)
    dest=a.output/'examples.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    load_instructions(dest)
    manifest=dict(source=SOURCE,revision=REVISION,train_sha256=TRAIN_SHA256,license='CC-BY-4.0',
                  base_sha256=base_hash,dataset_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  counts=dict(counts),splits=dict(Counter(r['split'] for r in rows)),
                  official_validation_downloaded=False,external_weights_used=False,
                  note='Reserva por título dentro do treino oficial; contextos inteiros, sem truncamento. Replay autoral inclui conflitos e ausência; QA não vira rótulo de requisito atendido. Desenvolvimento, sem promoção.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
