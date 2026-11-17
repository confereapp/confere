"""Teste único após congelamento: relações, atalhos e regressão local separadas."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from confere_ai.evidence_classifier import EvidencePredictor
from confere_ai.instruct import load_instructions
from confere_ai.relation_partition import words
from probe_conditioned_evidence import load_probe
from train_evidence_classifier import score

STATES = ['atendido', 'nao_atendido', 'evidencia_insuficiente']


def require_validation_gate(rows,run_summary):
    if any(r['provenance'].get('relation_matrix_version')==2 for r in rows):
        if run_summary.get('heldout_eligible') is not True:
            raise ValueError('Reserva da matriz preservada: candidato não passou a validação e retenção.')


def shortcut_baseline(train, test, side):
    """NB simples; vocabulário e contagens exclusivamente do treino."""
    def tokens(row):
        text = row['requirement'] if side == 'query' else ' '.join(u['text'] for u in row['units'])
        return words(text)
    counts = {s: Counter() for s in STATES}; totals = Counter(); classes = Counter(); vocabulary = set()
    for row in train:
        label = json.loads(row['target'])['status']; terms = tokens(row)
        counts[label].update(terms); totals[label] += len(terms); classes[label] += 1; vocabulary.update(terms)
    predictions = []
    for row in test:
        terms = [t for t in tokens(row) if t in vocabulary]
        scores = {s: math.log((classes[s]+1)/(len(train)+len(STATES)))
                  + sum(math.log((counts[s][t]+1)/(totals[s]+len(vocabulary))) for t in terms) for s in STATES}
        predictions.append(max(scores,key=scores.get))
    return predictions


def summarize(rows, predictions, exact_matches=None):
    if len(rows)!=len(predictions) or (exact_matches is not None and len(rows)!=len(exact_matches)):
        raise ValueError('Uma decisão é exigida por exemplo.')
    groups = defaultdict(lambda: dict(correct=0,total=0,false_approval=0))
    for index,(row, actual) in enumerate(zip(rows,predictions)):
        expected = json.loads(row['target'])['status']; g = groups[row['evaluation_profile']]
        g['correct'] += exact_matches[index] if exact_matches is not None else actual == expected
        g['total'] += 1
        g['false_approval'] += actual == 'atendido' and expected != 'atendido'
    return dict(groups)


def query_groups(rows, samples):
    if len(rows)!=len(samples):raise ValueError('Uma amostra é exigida por exemplo.')
    groups=defaultdict(list)
    for row,sample in zip(rows,samples):
        if row['provenance']['kind']=='public_paragraph_criterion':
            assert row['id']==sample['id']
            groups[row['source_document'],row['requirement']].append((row,sample))
    informative=[values for values in groups.values() if len({r['target'] for r,s in values})>1]
    return dict(total=len(groups),informative=len(informative),
                all_pairs_correct=sum(all(s['exact_match'] for r,s in values) for values in informative))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','probe','legacy-data','output']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps'); a=p.parse_args()
    if a.output.exists(): p.error('Não sobrescrever teste já executado.')
    torch.set_num_threads(4)
    freeze = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.data,a.probe,a.legacy_data,a.probe.parent/'source.pt']}
    rows,_=load_instructions(a.data);train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
    run_summary=json.loads((a.probe.parent/'evaluation.json').read_text())
    require_validation_gate(rows,run_summary)
    a.output.mkdir(parents=True)
    (a.output/'freeze.json').write_text(json.dumps(freeze,indent=2))
    predictor=load_probe(a.probe,a.device)
    baseline=score(predictor.encoder.base.model,[],test,a.device,predictor.encoder.base.thresholds,predict_units=predictor.encoder.base.predict_units)
    result=score(predictor.head,[],test,'cpu',predictor.encoder.base.thresholds,predict_units=predictor.predict_units)
    legacy,_=load_instructions(a.legacy_data);legacy=[r for r in legacy if r['split']=='validation']
    # A regressão usa o protocolo original de sentenças; não mistura sua métrica com parágrafos.
    original_state=torch.load(a.probe.parent/'source.pt',map_location='cpu',weights_only=True)
    original_state['settings']=dict(original_state['settings'],unit_segmentation='sentences_v1')
    native=EvidencePredictor(original_state,a.device)
    from probe_conditioned_evidence import ConditionedPredictor, FrozenPairs
    legacy_probe=ConditionedPredictor(FrozenPairs(original_state,a.device),predictor.head)
    legacy_base=score(native.model,[],legacy,a.device,native.thresholds,predict_units=native.predict_units)
    legacy_result=score(predictor.head,[],legacy,'cpu',native.thresholds,predict_units=legacy_probe.predict_units)
    predictions=[json.loads(r['actual'])['status'] for r in result['samples']]
    source_predictions=[json.loads(r['actual'])['status'] for r in baseline['samples']]
    queries=shortcut_baseline(train,test,'query'); bodies=shortcut_baseline(train,test,'body')
    diagnostics = dict(source=summarize(test,source_predictions,[r['exact_match'] for r in baseline['samples']]),
                       candidate=summarize(test,predictions,[r['exact_match'] for r in result['samples']]),
                       query_only=summarize(test,queries),body_only=summarize(test,bodies))
    query_control=query_groups(test,result['samples'])
    (a.output/'test.json').write_text(json.dumps(dict(baseline=baseline,candidate=result,diagnostics=diagnostics),ensure_ascii=False))
    (a.output/'legacy.json').write_text(json.dumps(dict(baseline=legacy_base,candidate=legacy_result),ensure_ascii=False))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in freeze.items())
    public=[r for r in test if r['provenance']['kind']=='public_reviewed_nli']
    local=[r for r in test if r['provenance']['kind']=='public_paragraph_criterion']
    strict=all(g['correct']/g['total']>=.9 and not g['false_approval'] for g in diagnostics['candidate'].values())
    retention=all(not b['exact_match'] or c['exact_match'] for b,c in zip(legacy_base['samples'],legacy_result['samples']))
    shortcut_gate={}
    for domain in ['public_nli_', 'academic_paragraph_']:
        totals={name:sum(g['correct'] for profile,g in profiles.items() if profile.startswith(domain))
                for name,profiles in diagnostics.items()}
        count=sum(g['total'] for profile,g in diagnostics['candidate'].items() if profile.startswith(domain))
        shortcut_gate[domain]=dict(correct=totals,total=count,
            passed=all(totals['candidate']-totals[name]>=max(1,.05*count) for name in ['query_only','body_only']))
    summary=dict(freeze=freeze,diagnostics=diagnostics,public_test_rows=len(public),editorial_test_rows=len(local),
                 legacy=dict(baseline_correct=legacy_base['correct'],candidate_correct=legacy_result['correct'],total=len(legacy),no_case_regression=retention),
                 strict_relation_gate_passed=strict,shortcut_gate=shortcut_gate,query_group_control=query_control,
                 gates_passed=strict and retention and all(g['passed'] for g in shortcut_gate.values())
                     and query_control['informative']>=1 and query_control['all_pairs_correct']/query_control['informative']>=.9,
                 adopted=False,promotable=False,test_selection_used=False,
                 note='Teste agrupado interno executado uma vez após seleção por validação. Se orientar nova iteração, passa a desenvolvimento. Anotações editoriais sem revisão humana independente. Controles NB são diagnósticos, não prova de compreensão. Não promove nem testa HTTP.')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
