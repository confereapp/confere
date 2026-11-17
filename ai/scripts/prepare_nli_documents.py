"""Mescla somente o treino público de inferência, preservando a validação local."""
import argparse
import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from confere_ai.evidence_classifier import examples, sentence_parts
from confere_ai.grounding import judge, prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.nli_source import SOURCE, REVISION, TRAIN_SHA256, LICENSE_SHA256, STATES
from confere_ai.tokenizer import ByteBPE


def annotated_row(item):
    label = int(item['label'])
    status = STATES[label]
    unit = dict(id='page:1', text=item['premise'].strip())
    requirement = item['hypothesis'].strip()
    evidence = [] if label == 2 else [dict(status=status, unit=unit['id'])]
    key = 'inferbr-train-' + item['sentence_pair_id']
    row = dict(id=key, project_id=key, split='train', task='grounded_' + status,
               grounding_schema=2, evaluation_profile='public_nli_train', requirement=requirement,
               units=[unit], document=key+'.txt', prompt=prompt_for(requirement, [unit], 2),
               target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
               provenance=dict(kind='public_reviewed_nli', source=SOURCE, revision=REVISION,
                               license='MIT', train_sha256=TRAIN_SHA256, license_sha256=LICENSE_SHA256,
                               source_row_id=item['sentence_pair_id'], source_label=label,
                               note='Geração semiautomática/automática revisada na origem; sem revisão humana local.'))
    assert judge(row['target'], row)['exact_match']
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'base', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Saída deve ser nova.')
    raw = (a.source/'train.csv').read_bytes()
    license_raw = (a.source/'LICENSE').read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256 or hashlib.sha256(license_raw).hexdigest() != LICENSE_SHA256:
        p.error('Snapshot ou licença não corresponde à origem auditada.')
    source = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = ByteBPE(source['merges'])
    rows, base_hash = load_instructions(a.base)
    if any(not r['task'].startswith('grounded_') for r in rows): p.error('Base documental exigida.')
    original_validation = [r for r in rows if r['split'] != 'train']
    seen = {r['prompt'].strip().casefold() for r in rows}
    # Exclui premissas e hipóteses da avaliação, além de pedidos completos iguais.
    reserved = {t.strip().casefold() for r in original_validation for t in [r['requirement']]+[u['text'] for u in r['units']]}
    counts = Counter()
    items = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    assert len(items) == 8190
    for item in items:
        if any(not item[k].strip() for k in ('premise','hypothesis','sentence_pair_id')): raise ValueError('Par incompleto.')
        row = annotated_row(item)
        if any(t.strip().casefold() in reserved for t in (item['premise'],item['hypothesis'])):
            counts['evaluation_overlap'] += 1; continue
        if len(sentence_parts(item['premise'])) != 1:
            counts['multiple_sentences'] += 1; continue
        key = row['prompt'].strip().casefold()
        if key in seen: counts['duplicate'] += 1; continue
        try:
            examples([row], tokenizer, source['config']['block_size'], readout=source['settings']['readout'])
        except ValueError:
            counts['too_long'] += 1; continue
        seen.add(key); rows.append(row); counts['added_'+str(item['label'])] += 1
    assert all(counts['added_'+str(label)] for label in STATES)
    assert [r for r in rows if r['split'] != 'train'] == original_validation
    a.output.mkdir(parents=True)
    target = a.output/'examples.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    load_instructions(target)
    shutil.copyfile(a.source/'LICENSE', a.output/'InferBR-LICENSE')
    manifest = dict(source=SOURCE, revision=REVISION, train_sha256=TRAIN_SHA256, license_sha256=LICENSE_SHA256,
                    base_sha256=base_hash, dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    counts=dict(counts), splits=dict(Counter(r['split'] for r in rows)), label_mapping=STATES,
                    test_used=False, official_validation_used=False, local_validation_unchanged=True,
                    note='Inferência textual como supervisão semântica auxiliar; não são documentos de projetos reais. Só premissas de uma sentença para alinhar à segmentação. Sem pesos externos.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
