"""Compara greedy e probabilidade de JSON completo em coorte estratificada de desenvolvimento."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.grounding import judge
from confere_ai.instruct import load_instructions
from confere_ai.model import GPT, GPTConfig
from confere_ai.project_review import input_ids
from confere_ai.structured_probe import choose
from confere_ai.tokenizer import ByteBPE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--per-profile",type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() or args.per_profile<1:parser.error("Saída deve ser nova e limite positivo.")
    info=json.loads((args.run_dir/'run.json').read_text())
    evaluation=json.loads((args.run_dir/'instruction-evaluation.json').read_text())
    audit=json.loads((args.run_dir/'context-audit.json').read_text())
    checkpoint_path=args.run_dir/'best.pt'
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==audit['checkpoint_sha256']
    rows,digest=load_instructions(Path(info['data']));assert digest==info['dataset_fingerprint']
    reference={r['id']:r for r in evaluation['selected']['samples']}
    selected=[];counts=Counter()
    for row in rows:
        if row['split']=='validation' and row['task'].startswith('grounded_'):
            profile=row.get('evaluation_profile','default')
            if counts[profile]<args.per_profile:selected.append(row);counts[profile]+=1
    torch.set_num_threads(4)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=GPT(GPTConfig(**checkpoint['config'])).to('mps');model.load_state_dict(checkpoint['model']);model.eval()
    tokenizer=ByteBPE(checkpoint['merges']);results=[]
    for row in selected:
        old=reference[row['id']];assert (old['prompt'],old['expected'])==(row['prompt'],row['target'])
        prefix=input_ids(row['requirement'],row['units'],tokenizer,2)
        response=choose(model,tokenizer,prefix,[u['id'] for u in row['units']])
        result=dict(id=row['id'],profile=row.get('evaluation_profile'),expected=row['target'],greedy=old['actual'],candidate=response,
                    greedy_correct=old['exact_match'],candidate_judgment=judge(response,row))
        results.append(result);print(json.dumps(dict(id=row['id'],greedy=old['exact_match'],candidate=result['candidate_judgment']['exact_match'])),flush=True)
    output=dict(checkpoint_sha256=audit['checkpoint_sha256'],method='Full response sum negative log probability, including EOS, no length normalization; finite schema V2 candidates in canonical alias/status order. Not all equivalent serializations are enumerated.',
                greedy_correct=sum(r['greedy_correct'] for r in results),candidate_correct=sum(r['candidate_judgment']['exact_match'] for r in results),
                total=len(results),results=results,adopted=False,note='Experimento offline estratificado; não altera geração nem decisões da API. Métrica de desenvolvimento, sem consulta a teste reservado.')
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
