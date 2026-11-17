"""Avaliação local de PDFs reais candidatos; sem gradientes, upload ou promoção."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from confere_ai.extract import extract_document
from confere_ai.project_review import review_documents


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, documents, candidates):
    sources={(f'doc:{i}',u['id']):(d,u) for i,d in enumerate(documents,1) for u in d['units']}
    assert len(result['requirements'])==len(candidates)
    rows=[]
    for actual,case in zip(result['requirements'],candidates):
        assert actual['requirement']==case['requirement']['text']
        spans=defaultdict(list)
        for call in actual['calls']:
            for part in call['sources']:
                key=(part['document_id'],part['source_unit']);doc,unit=sources[key]
                assert part['source_sha256']==doc['sha256']
                spans[key].append((part['char_start'],part['char_end']))
        for key,(doc,unit) in sources.items():
            if not unit['text'].strip(' |'): continue
            intervals=spans[key]
            assert intervals and intervals[0][0]==0 and intervals[-1][1]==len(unit['text'])
            assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        for evidence in actual['evidence']:
            doc,unit=sources[evidence['document_id'],evidence['source_unit']]
            assert evidence['source_sha256']==doc['sha256']
            assert evidence['quote']==unit['text'][evidence['char_start']:evidence['char_end']]
        expected=case['expected']['status']
        # shortcut: âncoras por citação individual; trechos divididos entre lotes exigem revisão manual.
        anchors=[any(e['source_unit']==gold['unit'] and e['model_status']==expected
                     and ' '.join(gold['quote'].split()) in ' '.join(e['quote'].split())
                     for e in actual['evidence']) for gold in case['expected']['evidence']]
        rows.append(dict(id=case['id'],requirement=actual['requirement'],expected=expected,actual=actual['status'],
                         status_match=actual['status']==expected,
                         false_positive=actual['status']=='atendido' and expected!='atendido',
                         valid=actual['model_outputs_valid'],conflict=actual['conflicting_evidence'],
                         missing_text_units=actual['coverage']['units_without_text'],
                         batches=len(actual['calls']),evidence_count=len(actual['evidence']),
                         annotated_quote_hits=sum(anchors),annotated_quotes=len(anchors),
                         note='Coincidência de status e âncoras são métricas separadas; citações adicionais não têm semântica auditada.'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    a.output.mkdir(parents=True)
    checkpoint_hash=digest(a.checkpoint)
    prepared=[];freeze=[]
    for path in a.candidates:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows and all(r['provenance']['kind']=='real' for r in rows)
        names=[n for n in rows[0]['documents'] if n!=rows[0]['requirement']['source']['document']]
        assert names and all([n for n in r['documents'] if n!=r['requirement']['source']['document']]==names for r in rows)
        documents=[extract_document(path.parent/n) for n in names]
        available={u['id']:' '.join(u['text'].split()) for d in documents for u in d['units']}
        annotation_checks=[dict(id=r['id'],quotes_found=[(' '.join(e['quote'].split()) in available[e['unit']])
                           for e in r['expected']['evidence']]) for r in rows]
        prepared.append((path,rows,documents,annotation_checks))
        freeze.append(dict(candidates=str(path),candidates_sha256=digest(path),
                           sources=[dict(path=str(path.parent/n),sha256=d['sha256']) for n,d in zip(names,documents)]))
    (a.output/'freeze.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_hash,
        inputs=freeze,training_executed=False,cohort_kind='known_real_development',script_sha256=digest(Path(__file__))),indent=2))
    summaries=[]
    for path,rows,documents,checks in prepared:
        started=time.perf_counter()
        result=review_documents(documents,[r['requirement']['text'] for r in rows],a.checkpoint,a.device)
        assert result['model']['sha256']==checkpoint_hash==digest(a.checkpoint)
        judgments=audit(result,documents,rows)
        name=path.parent.name
        payload=dict(project=name,annotation_checks=checks,judgments=judgments,trace=result)
        (a.output/(name+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        summary=dict(project=name,pages=sum(len(d['units']) for d in documents),
                     text_characters=sum(len(u['text']) for d in documents for u in d['units']),
                     requirements=len(rows),status_matches=sum(r['status_match'] for r in judgments),
                     false_positive=sum(r['false_positive'] for r in judgments),
                     invalid_requirements=sum(not r['valid'] for r in judgments),
                     conflicts=sum(r['conflict'] for r in judgments),
                     blocked_by_missing_text=sum(bool(r['missing_text_units']) for r in judgments),
                     batches=sum(r['batches'] for r in judgments),elapsed_seconds=time.perf_counter()-started,
                     contiguous_coverage_verified=True,literal_quotes_verified=True,
                     annotation_checks=checks,judgments=judgments)
        summaries.append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in {'judgments','annotation_checks'}},ensure_ascii=False),flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=checkpoint_hash,projects=summaries,
        training_executed=False,adopted=False,independent_human_review=False,
        note='PDFs reais conhecidos com OCR; checklist autoral e rótulos editoriais, não rubrica oficial. Status correto isolado não prova evidência correta. Sem promoção ou avaliação HTTP.'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
