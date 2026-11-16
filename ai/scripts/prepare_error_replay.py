"""Repetir apenas erros de primeiro token no treino, preservando avaliação e conversas."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.instruct import load_instructions
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE
from confere_ai.train import batch
from confere_ai.training_data import encode_sample


def replay(rows, errors, copies=16):
    projects = {r['project_id'] for r in rows if r['id'] in errors and r['split']=='train'}
    selected = [r for r in rows if r['project_id'] in projects]
    if any(r['split']!='train' for r in selected):
        raise ValueError('Replay atravessa splits.')
    added=[]
    for i in range(copies):
        for row in selected:
            added.append(dict(row, id=f'error-replay-{i}-{row["id"]}',
                              project_id=f'error-replay-{i}-{row["project_id"]}'))
    return rows+added


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','checkpoint','output']: p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Saída já existe.')
    torch.set_num_threads(4)
    rows,digest=load_instructions(a.data)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(source['merges']);model=GPT(GPTConfig(**source['config'])).to('mps')
    model.load_state_dict(source['model']);model.eval().requires_grad_(False)
    train=[r for r in rows if r['split']=='train']
    encoded=[encode_sample(r,tokenizer,model.config.block_size) for r in train]
    errors=[];counts=Counter()
    for offset in range(0,len(train),8):
        x,y=batch(encoded,list(range(offset,min(offset+8,len(train)))),'mps',model.config.block_size)
        positions=(y!=-100).long().argmax(-1);indices=torch.arange(len(x),device='mps')
        with torch.no_grad(): predicted=model(x)[indices,positions].argmax(-1)
        wrong=(predicted!=y[indices,positions]).cpu().tolist()
        for j,bad in enumerate(wrong):
            if bad: errors.append(train[offset+j]['id']);counts[train[offset+j]['task']]+=1
        if offset%8000==0: print(json.dumps(dict(examined=offset,errors=len(errors))),flush=True)
    result=replay(rows,set(errors))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    _,new_digest=load_instructions(a.output)
    assert [r for r in result if r['split']!='train']==[r for r in rows if r['split']!='train']
    manifest=dict(source=str(a.data),source_sha256=digest,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),output_sha256=new_digest,errors=errors,families=dict(counts),train_examples=len(train),added=len(result)-len(rows),copies=16,selection='train_first_response_token_argmax_error',test_examples_used=0)
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(families=dict(counts),added=manifest['added'])),flush=True)


if __name__=='__main__': main()
