"""Ajuste de prior de saída: somente viés final, dados de treino, validação por geração."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.chat import prepare_prompt
from confere_ai.evaluation_cache import protocol_fingerprint
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import load_instructions, score, acceptance, conversation_indices
from confere_ai.model import GPT,GPTConfig
from confere_ai.tokenizer import ByteBPE
from confere_ai.train import batch


def fit_prior(logits, weights, labels):
    logits=logits.detach();weights=weights.detach();delta=torch.nn.Parameter(torch.zeros(weights.shape[1]))
    optimizer=torch.optim.LBFGS([delta],lr=1.,max_iter=80,line_search_fn="strong_wolfe")
    def objective():return torch.nn.functional.cross_entropy(logits+delta@weights.T,labels)+.001*delta.square().mean()
    initial=objective().item()
    def closure():
        optimizer.zero_grad(set_to_none=True);loss=objective()
        if not torch.isfinite(loss):raise ValueError("Loss não finita.")
        loss.backward();return loss
    optimizer.step(closure)
    return delta.detach(),dict(initial_loss=initial,final_loss=objective().item(),iterations=optimizer.state[delta]["n_iter"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint","data","reference","run-dir"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.run_dir.exists():p.error("Saída já existe.")
    torch.set_num_threads(4);torch.manual_seed(42)
    rows,digest=load_instructions(args.data);train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if any(not r['task'].startswith('conversation_') for r in rows):p.error("Somente conversa.")
    raw=args.checkpoint.read_bytes();source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=ByteBPE(source['merges']);model=GPT(GPTConfig(**source['config'])).to('mps');model.load_state_dict(source['model']);model.eval().requires_grad_(False)
    pools={'legacy':{},'context':{}}
    for i,row in enumerate(train):
        category='context' if row['task'].startswith('conversation_context_answer_') else 'legacy'
        pools[category].setdefault(row['task'],[]).append(i)
    if not all(pools.values()):p.error('Exige ambos os grupos de conversa.')
    indices=conversation_indices(pools,3200,torch.Generator().manual_seed(43));encoded=[]
    for i in indices:
        row=train[i];assert row['split']=='train'
        ids,dropped=prepare_prompt(row['history'],row['prompt'],tokenizer,1024,256,'conversation');assert dropped==0
        target=tokenizer.encode(row['target'])[0];encoded.append((ids,[-100]*(len(ids)-1)+[target]))
    chunks=[];targets=[]
    for offset in range(0,len(encoded),8):
        x,y=batch(encoded,list(range(offset,min(offset+8,len(encoded)))),'mps',1024)
        positions=(y!=-100).long().argmax(-1)
        with torch.no_grad():chunks.append(model(x)[torch.arange(len(x),device='mps'),positions].cpu())
        targets.extend(y[torch.arange(len(x),device='mps'),positions].cpu().tolist())
    logits=torch.cat(chunks);labels=torch.tensor(targets);delta,fit=fit_prior(logits,model.head.weight.cpu(),labels)
    args.run_dir.mkdir(parents=True);(args.run_dir/'source.pt').write_bytes(raw)
    torch.save(dict(delta=delta,train_ids=[train[i]['id'] for i in indices],logits=logits,labels=labels),args.run_dir/'fit.pt')
    reference=torch.load(args.reference,map_location='cpu',weights_only=True)
    if reference['config']!=source['config'] or reference['merges']!=source['merges']:raise ValueError('Referência incompatível.')
    ref=GPT(GPTConfig(**reference['config'])).to('mps');ref.load_state_dict(reference['model']);ref.eval();baseline=score(ref,tokenizer,validation,'mps',1024,256);del ref
    base_bias=model.norm.bias.detach().clone();best=selected=alpha_selected=None
    settings=dict(source['settings'],objective='conversation',generation_budget=256,trainable_scope='final_normalization_bias',output_prior_fit='train_first_token_lbfgs_v1')
    for alpha in [0.,.25,.5,.75,1.,1.5,2.]:
        with torch.no_grad():model.norm.bias.copy_(base_bias+alpha*delta.to('mps'))
        result=score(model,tokenizer,validation,'mps',1024,256);key=quality_key(result)
        print(json.dumps(dict(alpha=alpha,correct=result['correct'],total=result['total'],quality=key)),flush=True)
        (args.run_dir/f'candidate-{alpha}.json').write_text(json.dumps(dict(result=result,quality=key),ensure_ascii=False))
        if best is None or key>best:
            best,selected,alpha_selected=key,result,alpha
            weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
            assert all(torch.equal(v,source['model'][k]) for k,v in weights.items() if k!='norm.bias')
            checkpoint={k:v for k,v in source.items() if k not in ['optimizer','generator_state']}
            checkpoint.update(model=weights,dataset_fingerprint=digest,settings=dict(settings,output_prior_alpha=alpha))
            torch.save(checkpoint,args.run_dir/'best.pt')
    info=dict(objective='conversation',data=str(args.data),dataset_fingerprint=digest,device='mps',paired_evaluation_context=1024,paired_evaluation_response_tokens=256,evaluation_protocol_hash=protocol_fingerprint(),settings=settings,source=str(args.checkpoint),source_sha256=hashlib.sha256(raw).hexdigest(),baseline_checkpoint=str(args.reference),source_step=source['step'],trainable_parameters=len(delta),sampled_train_examples=len(indices),sampled_families=dict(Counter(train[i]['task'] for i in indices)),test_examples_used=0,fit=fit,note='Somente viés final ajustado em exemplos de treino. Alphas selecionados em desenvolvimento; nenhum pedido completo dos diagnósticos no treino. Novo teste independente obrigatório.')
    (args.run_dir/'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    (args.run_dir/'instruction-evaluation.json').write_text(json.dumps(dict(baseline=baseline,selected=selected,selected_step=source['step'],alpha=alpha_selected,quality=best,acceptance=acceptance(selected),test_examples_used=0,adopted=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
