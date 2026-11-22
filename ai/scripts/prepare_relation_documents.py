"""Corpus de relações com fontes fixas e separação por documentos/componentes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import torch
from confere_ai.evidence_classifier import examples
from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.nli_source import TRAIN_SHA256 as NLI_SHA, LICENSE_SHA256, STATES
from confere_ai.qa_source import SOURCE, REVISION, TRAIN_SHA256 as QA_SHA
from confere_ai.relation_partition import components, split_for, PROTOCOL, verify_partition
from confere_ai.tokenizer import ByteBPE
from prepare_nli_documents import annotated_row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paragraph_rows(source, annotations):
    documents = {}
    for document in source['data']:
        documents.setdefault(document['title'], []).extend(document['paragraphs'])
    rows = []
    for item in annotations['documents']:
        title, index = item['title'], item['paragraph_index']
        text = documents[title][index]['context']
        anchor = item['anchor']; start = text.index(anchor)
        for claim in item['claims']:
            label = claim['label']; status = STATES[label]
            key = f"paragraph-{title}-{index}-{claim.get('id',label)}"
            units = [dict(id='page:1', text=text)]
            evidence = [] if label == 2 else [dict(status=status, unit='page:1')]
            rows.append(dict(id=key, project_id='paragraph-'+title, source_document='faquad:'+title,
                split=item['split'], task='grounded_'+status, grounding_schema=2,
                evaluation_profile='academic_paragraph_'+status, requirement=claim['text'], units=units,
                document='paragraph-'+title+'.txt', prompt=prompt_for(claim['text'], units, 2),
                target=json.dumps(dict(status=status,evidence=evidence),ensure_ascii=False),
                provenance=dict(kind='public_paragraph_criterion',source=SOURCE,revision=REVISION,
                    train_sha256=QA_SHA,license='CC-BY-4.0',partition_protocol=PROTOCOL,
                    annotation_kind='local_editorial',independent_human_review=False,
                    paragraph_sha256=hashlib.sha256(text.encode()).hexdigest(),paragraph_index=index,
                    anchor=anchor,anchor_start=start,rationale=claim['rationale'])))
    return rows


def assign_partitions(rows):
    keys = components(rows)
    assigned = {}
    priority = {'train': 0, 'validation': 1, 'test': 2}
    for row in rows:
        group = keys[row['id']]
        split = row['split'] if row.get('source_document') else split_for(group)
        if group not in assigned or priority[split] > priority[assigned[group]]: assigned[group] = split
    for row in rows:
        group = keys[row['id']]
        row['split'] = assigned[group]
        row['project_id'] = group
        row['provenance']['partition_protocol'] = PROTOCOL
        row['provenance']['component_sha256'] = group
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['nli-source', 'qa-source', 'annotations', 'checkpoint', 'output']:
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Saída deve ser nova.')
    if sha(a.nli_source/'train.csv') != NLI_SHA or sha(a.nli_source/'LICENSE') != LICENSE_SHA256 or sha(a.qa_source) != QA_SHA:
        p.error('Snapshot ou licença alterado.')
    source = torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer = ByteBPE(source['merges'])
    records = [annotated_row(item) for item in csv.DictReader((a.nli_source/'train.csv').open())]
    for row in records: row['evaluation_profile'] = 'public_nli_' + row['task'].removeprefix('grounded_')
    records += paragraph_rows(json.loads(a.qa_source.read_text()),json.loads(a.annotations.read_text()))
    # Deduplicação antes do split, sem usar previsões nem os rótulos para escolher grupos.
    unique = {}; duplicates = 0
    for row in records:
        key = row['prompt'].strip().casefold()
        if key in unique:
            if unique[key]['target'] != row['target']: raise ValueError('Par com alvos conflitantes.')
            duplicates += 1
        else: unique[key] = row
    records = assign_partitions(list(unique.values()))
    retained = []; rejected = Counter()
    for row in records:
        try: examples([row],tokenizer,source['config']['block_size'],readout=source['settings']['readout'])
        except ValueError: rejected[row['split']+':too_long'] += 1; continue
        retained.append(row)
    verify_partition(retained)
    assert all(any(r['split']==s for r in retained) for s in ['train','validation','test'])
    assert all(any(r['split']==s and r['provenance']['kind']=='public_paragraph_criterion' for r in retained) for s in ['train','validation','test'])
    a.output.mkdir(parents=True)
    path = a.output/'examples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in retained))
    load_instructions(path)
    manifest = dict(protocol=PROTOCOL,dataset_sha256=sha(path),annotations_sha256=sha(a.annotations),
        nli_train_sha256=NLI_SHA,qa_train_sha256=QA_SHA,source_checkpoint_sha256=sha(a.checkpoint),
        counts=dict(Counter(r['split']+':'+r['evaluation_profile'] for r in retained)),
        documents={s:sorted({r['source_document'] for r in retained if r['split']==s and r.get('source_document')}) for s in ['train','validation','test']},
        components=len({r['project_id'] for r in retained}),duplicates_removed=duplicates,rejected=dict(rejected),
        label_mapping=STATES,split_uses_labels=False,near_duplicate_rule='Jaccard de conjuntos de tokens >= .8 em qualquer lado; componentes transitivos; títulos inteiros agrupados',
        official_test_used=False,tccs_used=False,independent_human_review=False,
        limitations=['Similaridade lexical não detecta toda paráfrase.','Anotações locais editoriais não possuem segunda revisão humana.','Inferência pública é auxiliar; não comprova qualidade em TCCs.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
