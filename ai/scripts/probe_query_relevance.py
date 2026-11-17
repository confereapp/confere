"""Filtro lexical experimental antes da classificação; não altera API ou pesos."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import torch
from confere_ai.evidence_classifier import EvidencePredictor, sentence_parts, response
from confere_ai.extract import extract_document
from confere_ai.project_review import review_extracted
from evaluate_real_documents import audit, digest

STOP=set('a ao aos as o os um uma uns umas de do dos da das em no nos na nas por para com sem e ou que se seu sua seus suas este esta estes estas esse essa esses essas foi foram ser estar deve devem precisa precisam deverá relatório relatorio documento projeto texto solução solucao descrita incluir apresentar descrever identificar informar demonstrar registrar comprovar oferecer conter ter há ha existe existência existencia presença presenca ausência ausencia implementação implementacao'.split())


def terms(text):
    plain=''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))
    words=re.findall(r'[^\W\d_]+',plain)
    return {w[:-1] if len(w)>3 and w.endswith('s') else w for w in words if len(w)>2 and w not in STOP}


def candidate(query, text):
    query_terms=terms(query)
    # shortcut: sem termos úteis o filtro não interfere; sinônimos sem termo comum podem ser perdidos.
    return not query_terms or bool(query_terms & terms(text))


def predict(predictor, query, units):
    probabilities=[[0.,0.] for _ in units]
    kept=[(i,text) for i,u in enumerate(units) for text in sentence_parts(u['text']) if candidate(query,text)]
    for start in range(0,len(kept),4):
        group=kept[start:start+4]
        values=predictor.probabilities(query,[dict(id=f'page:{j+1}',text=text) for j,(_,text) in enumerate(group)])
        for (i,_),value in zip(group,values):
            probabilities[i]=[max(a,b) for a,b in zip(probabilities[i],value)]
    return response(units,probabilities,predictor.thresholds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Saída deve ser nova.')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    predictor=EvidencePredictor(payload,'mps')
    if predictor.segmentation!='sentences_v1':raise ValueError('Ensaio exige classificador por frase.')
    predictor.segmentation='whole_unit_v1'
    checkpoint_hash=digest(a.checkpoint);summaries=[]
    (a.output/'freeze.json').write_text(json.dumps(dict(checkpoint_sha256=checkpoint_hash,
        script_sha256=digest(Path(__file__)),candidates=[dict(path=str(x),sha256=digest(x)) for x in a.candidates],
        method='experimental_query_terms_v1',promotable=False),indent=2))
    for path in a.candidates:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        names=[n for n in rows[0]['documents'] if n!=rows[0]['requirement']['source']['document']]
        documents=[extract_document(path.parent/n) for n in names]
        result=review_extracted(documents,[r['requirement']['text'] for r in rows],predictor.tokenizer,
            predictor.config.block_size,None,2,256,predict_units=lambda q,u:predict(predictor,q,u))
        judgments=audit(result,documents,rows)
        summary=dict(project=path.parent.name,status_matches=sum(r['status_match'] for r in judgments),
            total=len(rows),false_positive=sum(r['false_positive'] for r in judgments),
            conflicts=sum(r['conflict'] for r in judgments),evidence_count=sum(r['evidence_count'] for r in judgments),
            blocked_by_missing_text=sum(bool(r['missing_text_units']) for r in judgments),judgments=judgments)
        (a.output/(path.parent.name+'.json')).write_text(json.dumps(dict(summary=summary,trace=result),ensure_ascii=False,indent=2))
        summaries.append(summary);print(json.dumps({k:v for k,v in summary.items() if k!='judgments'}),flush=True)
    assert digest(a.checkpoint)==checkpoint_hash
    (a.output/'summary.json').write_text(json.dumps(dict(method='experimental_query_terms_v1',projects=summaries,
        checkpoint_sha256=checkpoint_hash,promotable=False,training_executed=False,
        note='Filtro lexical de candidatos, não prova suporte semântico. Desenvolvimento conhecido; sem promoção ou mudança na API.'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
