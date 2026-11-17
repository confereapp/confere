"""Ajusta o encoder próprio para relevância/estado por unidade, sem decoder JSON."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from confere_ai.evidence_classifier import EvidenceClassifier, examples, unit_batch, response, READOUTS, paired_indices, unit_weighted_query_pools, STATE_MODES, ATTENTION_MODES, load_binary_as_categorical
from confere_ai.finetuning import configure_trainable
from confere_ai.generative_selection import quality_key
from confere_ai.grounding import judge
from confere_ai.instruct import load_instructions, acceptance
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


def validation_predictor(source, model, readout, device):
    from confere_ai.evidence_classifier import EvidencePredictor
    effective=dict(source,settings=dict(source['settings'],readout=readout))
    predictor=EvidencePredictor(effective,device)
    predictor.model=model
    return predictor


def query_sampling_view(items, rows):
    queries={row['id']:row['requirement'] for row in rows}
    return [dict(item,source_key=hashlib.sha256(queries[item['row_id']].encode()).hexdigest()) for item in items]


@torch.no_grad()
def training_errors(model, train, device, thresholds=(.5, .5)):
    response([], [], thresholds)
    cutoffs=torch.tensor(thresholds,device=device)
    model.eval(); errors=[]
    for offset in range(0,len(train),8):
        items=train[offset:offset+8]
        ids,lengths=unit_batch(items,device,model.backbone.config.block_size)
        predicted=model.evidence_scores(model(ids,lengths))>=cutoffs
        labels=torch.tensor([item['labels'] for item in items],device=device).bool()
        errors.extend(offset+i for i,bad in enumerate((predicted!=labels).any(-1).cpu().tolist()) if bad)
    return errors


@torch.no_grad()
def score(model, encoded, rows, device, thresholds=(.5, .5), *, predict_units=None):
    model.eval(); probabilities = []
    for offset in range(0, len(encoded) if predict_units is None else 0, 8):
        ids, lengths = unit_batch(encoded[offset:offset+8], device, model.backbone.config.block_size)
        probabilities.extend(model.evidence_scores(model(ids, lengths)).cpu().tolist())
    results = []; offset = 0
    for row in rows:
        count = len(row["units"])
        raw = (predict_units(row['requirement'],row['units']) if predict_units else
               response(row["units"], probabilities[offset:offset+count], thresholds)); offset += count
        results.append(dict(id=row["id"], task=row["task"], prompt=row["prompt"], expected=row["target"], actual=raw,
                            evaluation_profile=row.get("evaluation_profile", "default"), **judge(raw, row)))
    assert predict_units is not None or offset == len(probabilities)
    return dict(correct=sum(r["exact_match"] for r in results), total=len(results), samples=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "init-from", "run-dir"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--categorical-states", action="store_true")
    parser.add_argument("--require-perfect-validation", action="store_true")
    parser.add_argument("--paired-queries", action="store_true", help="Primeira unidade uniforme; segunda consulta da mesma fonte com rótulo diferente quando disponível.")
    parser.add_argument('--bidirectional-encoder',action='store_true')
    parser.add_argument('--train-embeddings',action='store_true')
    parser.add_argument('--paragraph-units',action='store_true')
    parser.add_argument('--pair-by-query',action='store_true')
    parser.add_argument("--error-replay", action="store_true", help="Metade de cada lote revisita erros medidos somente no treino inicial.")
    parser.add_argument("--readout", choices=READOUTS)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--selection-patience", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    if (args.run_dir.exists() or args.steps <= 0 or args.selection_patience <= 0 or args.eval_every <= 0
            or any(not math.isfinite(v) or v <= 0 for v in [args.encoder_learning_rate, args.head_learning_rate])): parser.error("Saída nova e passos positivos são obrigatórios.")
    torch.set_num_threads(4); torch.manual_seed(42)
    rows, digest = load_instructions(args.data)
    if any(not r["task"].startswith("grounded_") for r in rows): parser.error("Somente documentos controlados.")
    source = torch.load(args.init_from, map_location="cpu", weights_only=True)
    backbone = GPT(GPTConfig(**source["config"]))
    if "classifier" not in source: backbone.load_state_dict(source["model"])
    source_mode = source["settings"].get("state_mode", STATE_MODES[0])
    state_mode = STATE_MODES[1] if args.categorical_states else source_mode
    attention_mode=ATTENTION_MODES[1] if args.bidirectional_encoder else source['settings'].get('attention_mode',ATTENTION_MODES[0])
    model = EvidenceClassifier(backbone, state_mode, attention_mode)
    if "classifier" in source:
        if source["settings"].get("readout") not in READOUTS: parser.error("Representação incompatível.")
        if source_mode == STATE_MODES[0] and state_mode == STATE_MODES[1]: load_binary_as_categorical(model, source["classifier"])
        else: model.load_state_dict(source["classifier"])
    model.to(args.device)
    configure_trainable(model.backbone,None if args.train_embeddings else model.backbone.config.layers)
    tokenizer = ByteBPE(source["merges"])
    readout = args.readout or source["settings"].get("readout", READOUTS[0])
    train = examples([r for r in rows if r["split"] == "train"], tokenizer, backbone.config.block_size, readout=readout)
    validation = [r for r in rows if r["split"] == "validation"]
    val = examples(validation, tokenizer, backbone.config.block_size, readout=readout)
    sampling_items=query_sampling_view(train,rows) if args.pair_by_query else train
    paired_pools = unit_weighted_query_pools(sampling_items) if args.paired_queries or args.pair_by_query else []
    generator = torch.Generator().manual_seed(42)
    error_thresholds=source['settings'].get('thresholds',[.5,.5])
    hard = training_errors(model,train,args.device,error_thresholds) if args.error_replay else []
    if args.error_replay: print(json.dumps(dict(train_error_units=len(hard),train_units=len(train))),flush=True)
    optimizer = torch.optim.AdamW([dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=args.encoder_learning_rate),
                                  dict(params=model.head.parameters(), lr=args.head_learning_rate)], weight_decay=.01)
    settings = dict(objective="document_classifier", state_mode=state_mode, readout=readout, frozen_embeddings=not args.train_embeddings, require_perfect_validation=args.require_perfect_validation,
                    encoder_learning_rate=args.encoder_learning_rate, head_learning_rate=args.head_learning_rate, selection_patience=args.selection_patience, batch_size=8, threshold=.5, eval_every=args.eval_every,
                    checkpoint_selection="generated_family_quality", sampling="paired_queries_uniform_unit_then_opposite_label" if args.paired_queries else "uniform_units", error_replay=args.error_replay, initial_train_errors=len(hard), loss="categorical_ce" if state_mode==STATE_MODES[1] else "unweighted_bce")
    settings['unit_segmentation']=source['settings'].get('unit_segmentation','whole_unit_v1')
    if args.paragraph_units:settings['unit_segmentation']='whole_unit_v1'
    if args.pair_by_query:settings['sampling']='paired_units_uniform_then_same_query_other_label'
    settings['attention_mode']=attention_mode
    settings['thresholds']=source['settings'].get('thresholds',[.5,.5])
    if args.error_replay:settings['training_error_thresholds']=list(error_thresholds)
    response([],[],settings['thresholds'])
    predictor=None
    if settings['unit_segmentation']=='sentences_v1':
        predictor=validation_predictor(source,model,readout,args.device)
    args.run_dir.mkdir(parents=True)
    info = dict(data=str(args.data), dataset_fingerprint=digest, source=str(args.init_from),
                source_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(), device=args.device,
                train_units=len(train), validation_units=len(val), settings=settings, optimizer_reset=True, test_examples_used=0,
                note="Predição semântica por unidade; JSON/referências por construção. Validação de desenvolvimento, sem teste cego; sem integração na API.")
    (args.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    if args.error_replay:
        (args.run_dir/'training-errors.json').write_text(json.dumps([dict(row_id=train[i]['row_id'],unit=train[i]['unit']) for i in hard]))
    best = baseline = selected = None; selected_step = None; stale = 0; started = time.perf_counter()
    def checkpoint(step, filename):
        temporary = args.run_dir / (filename + ".tmp")
        torch.save(dict(config=source["config"], merges=source["merges"], classifier={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        step=step, settings=settings, dataset_fingerprint=digest, optimizer=optimizer.state_dict(),
                        generator_state=generator.get_state()), temporary)
        temporary.replace(args.run_dir / filename)
    stop = "steps"
    for step in range(args.steps + 1):
        if step:
            model.train()
            indices = (paired_indices(sampling_items, paired_pools, 8, generator) if paired_pools
                       else torch.randint(len(train), (8,), generator=generator).tolist())
            if hard: indices[:4]=[hard[i] for i in torch.randint(len(hard),(4,),generator=generator).tolist()]
            items = [train[i] for i in indices]
            ids, lengths = unit_batch(items, args.device, backbone.config.block_size)
            labels = torch.tensor([item["labels"] for item in items], device=args.device)
            loss = model.loss(model(ids, lengths), labels)
            if not torch.isfinite(loss): raise ValueError("Loss não finita; checkpoint anterior preservado.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            checkpoint(step, "last.pt")
            result = score(model, val, validation, args.device,settings['thresholds'],predict_units=predictor.predict_units if predictor else None); key = quality_key(result)
            if baseline is None: baseline = result
            (args.run_dir / f"evaluation-{step}.json").write_text(json.dumps(dict(quality=key, result=result), ensure_ascii=False, indent=2))
            print(json.dumps(dict(step=step, correct=result["correct"], total=result["total"], quality=key,
                                  elapsed_seconds=time.perf_counter()-started)), flush=True)
            if best is None or key > best:
                best, selected, selected_step = key, result, step; stale = 0; checkpoint(step, "best.pt")
            else: stale += 1
            assert 0 <= result["correct"] <= result["total"]
            if key[0] and (not args.require_perfect_validation or result["correct"] == result["total"]):
                stop = "controlled_acceptance"; break
            if stale >= args.selection_patience: stop = "selection_plateau"; break
    (args.run_dir / "evaluation.json").write_text(json.dumps(dict(baseline=baseline, selected=selected, selected_step=selected_step,
        quality=best, acceptance=acceptance(selected), adopted=False), ensure_ascii=False, indent=2))
    (args.run_dir / "summary.json").write_text(json.dumps(dict(final_step=step, selected_step=selected_step, stop_reason=stop,
        elapsed_seconds=time.perf_counter()-started, test_examples_used=0), indent=2))


if __name__ == "__main__": main()
