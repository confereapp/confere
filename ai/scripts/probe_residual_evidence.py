"""Sonda não linear residual sobre encoder congelado; artefato não aceito pela API."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from confere_ai.evidence_classifier import EvidencePredictor, examples, unit_batch, STATE_MODES
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import load_instructions
from calibrate_evidence import evaluate_thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "data", "run-dir"]: p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--features-from", type=Path)
    args = p.parse_args()
    if args.run_dir.exists(): p.error("Saída já existe.")
    torch.set_num_threads(4);torch.manual_seed(42)
    payload=args.checkpoint.read_bytes();source=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    predictor=EvidencePredictor(source,"mps")
    if predictor.model.state_mode != STATE_MODES[0]:p.error("Sonda exige cabeça binária.")
    if predictor.segmentation != 'whole_unit_v1':p.error('Sonda exige representação por unidade inteira.')
    rows,digest=load_instructions(args.data);train_rows=[r for r in rows if r["split"]=="train"];val_rows=[r for r in rows if r["split"]=="validation"]
    if any(not r["task"].startswith("grounded_") for r in rows):p.error("Somente documentos controlados.")
    args.run_dir.mkdir(parents=True);(args.run_dir/"source.pt").write_bytes(payload)
    if args.features_from:
        previous=json.loads((args.features_from.parent/"run.json").read_text())
        if previous["dataset_fingerprint"] != digest or previous["source_sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("Cache não corresponde ao corpus e aos pesos congelados.")
        cached=torch.load(args.features_from,map_location="cpu",weights_only=True)
        x,y,v,base,val_base,mean,scale,indices=[cached[k] for k in ["train","labels","validation","base","val_base","mean","scale","train_sample_indices"]]
        assert len(x)==len(y)==len(base) and len(v)==sum(len(r["units"]) for r in val_rows)
    else:
        train=examples(train_rows,predictor.tokenizer,1024,readout=predictor.readout)
        indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(43))[:20000].tolist();train=[train[i] for i in indices]
        val=examples(val_rows,predictor.tokenizer,1024,readout=predictor.readout)
        def extract(items,label):
            chunks=[]
            for offset in range(0,len(items),16):
                ids,lengths=unit_batch(items[offset:offset+16],"mps",1024)
                with torch.no_grad():chunks.append(predictor.model.features(ids,lengths).cpu())
                if offset % 2048==0:print(json.dumps(dict(phase=label,units=offset,total=len(items))),flush=True)
            return torch.cat(chunks)
        x=extract(train,"train_features");v=extract(val,"validation_features")
        y=torch.tensor([item["labels"] for item in train]);head=predictor.model.head.cpu().requires_grad_(False)
        with torch.no_grad():base=head(x);val_base=head(v)
        mean=x.mean(0);scale=x.std(0).clamp_min(1e-4);x=(x-mean)/scale;v=(v-mean)/scale
    correction=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,2))
    nn.init.zeros_(correction[-1].weight);nn.init.zeros_(correction[-1].bias)
    assert torch.equal(val_base+correction(v),val_base)
    hard=torch.where(((base.sigmoid()>=.5).float()!=y).any(-1))[0]
    if not len(hard):hard=torch.arange(len(x))
    torch.save(dict(train=x,validation=v,labels=y,base=base,val_base=val_base,mean=mean,scale=scale,train_sample_indices=indices),args.run_dir/"features.pt")
    info=dict(method="frozen_encoder_residual_nonlinear_probe",data=str(args.data),dataset_fingerprint=digest,source_sha256=hashlib.sha256(payload).hexdigest(),
              train_units=len(x),validation_units=len(v),hard_train_units=len(hard),sampling="half_current_training_errors_half_uniform_refresh_500",hidden=128,
              training_only_normalization=True,test_examples_used=0,promotable=False,note="Sonda: nenhuma integração ou promoção automática. Seleção em desenvolvimento.")
    (args.run_dir/"run.json").write_text(json.dumps(info,ensure_ascii=False,indent=2))
    optimizer=torch.optim.AdamW(correction.parameters(),lr=.001,weight_decay=.01);generator=torch.Generator().manual_seed(44)
    best=selected=selected_step=None;stale=0;stop="steps"
    for step in range(5001):
        if step:
            if step % 500 == 0:
                with torch.no_grad():hard=torch.where((((base+correction(x)).sigmoid()>=.5).float()!=y).any(-1))[0]
                if not len(hard):hard=torch.arange(len(x))
            a=hard[torch.randint(len(hard),(64,),generator=generator)];b=torch.randint(len(x),(64,),generator=generator);batch=torch.cat([a,b])
            logits=base[batch]+correction(x[batch]);loss=nn.functional.binary_cross_entropy_with_logits(logits,y[batch])
            if not torch.isfinite(loss):raise ValueError("Loss não finita.")
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(correction.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step % 100==0:
            with torch.no_grad():probabilities=(val_base+correction(v)).sigmoid().tolist()
            result=evaluate_thresholds(val_rows,probabilities,[.5,.5]);key=quality_key(result)
            print(json.dumps(dict(step=step,correct=result["correct"],total=result["total"],quality=key)),flush=True)
            (args.run_dir/f"evaluation-{step}.json").write_text(json.dumps(dict(quality=key,result=result),ensure_ascii=False))
            if best is None or key>best:
                best,selected,selected_step=key,result,step;stale=0
                torch.save(dict(correction=correction.state_dict(),mean=mean,scale=scale,step=step,objective="residual_evidence_probe",source_sha256=info["source_sha256"]),args.run_dir/"correction.pt")
            else:stale+=1
            if result["correct"]==result["total"] and key[0]:stop="perfect_development";break
    (args.run_dir/"evaluation.json").write_text(json.dumps(dict(selected=selected,selected_step=selected_step,quality=best,adopted=False),ensure_ascii=False,indent=2))
    (args.run_dir/"summary.json").write_text(json.dumps(dict(final_step=step,selected_step=selected_step,stop_reason=stop,adopted=False),indent=2))


if __name__=="__main__":main()
