"""Ajusta o encoder próprio para relevância/estado por unidade, sem decoder JSON."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from confere_ai.evidence_classifier import EvidenceClassifier, examples, unit_batch, response, READOUTS, paired_indices
from confere_ai.finetuning import configure_trainable
from confere_ai.generative_selection import quality_key
from confere_ai.grounding import judge
from confere_ai.instruct import load_instructions, acceptance
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


@torch.no_grad()
def score(model, encoded, rows, device, thresholds=(.5, .5)):
    model.eval(); probabilities = []
    for offset in range(0, len(encoded), 8):
        ids, lengths = unit_batch(encoded[offset:offset+8], device, model.backbone.config.block_size)
        probabilities.extend(model(ids, lengths).sigmoid().cpu().tolist())
    results = []; offset = 0
    for row in rows:
        count = len(row["units"])
        raw = response(row["units"], probabilities[offset:offset+count], thresholds); offset += count
        results.append(dict(id=row["id"], task=row["task"], prompt=row["prompt"], expected=row["target"], actual=raw,
                            evaluation_profile=row.get("evaluation_profile", "default"), **judge(raw, row)))
    assert offset == len(probabilities)
    return dict(correct=sum(r["exact_match"] for r in results), total=len(results), samples=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "init-from", "run-dir"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--require-perfect-validation", action="store_true")
    parser.add_argument("--paired-queries", action="store_true")
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
    model = EvidenceClassifier(backbone)
    if "classifier" in source:
        if source["settings"].get("readout") not in READOUTS: parser.error("Representação incompatível.")
        model.load_state_dict(source["classifier"])
    model.to(args.device)
    configure_trainable(model.backbone, model.backbone.config.layers)
    tokenizer = ByteBPE(source["merges"])
    readout = args.readout or source["settings"].get("readout", READOUTS[0])
    train = examples([r for r in rows if r["split"] == "train"], tokenizer, backbone.config.block_size, readout=readout)
    validation = [r for r in rows if r["split"] == "validation"]
    val = examples(validation, tokenizer, backbone.config.block_size, readout=readout)
    pools = defaultdict(list)
    if args.paired_queries:
        for i, item in enumerate(train): pools[item["source_key"]].append(i)
    paired_pools = list(pools.values())
    generator = torch.Generator().manual_seed(42)
    optimizer = torch.optim.AdamW([dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=args.encoder_learning_rate),
                                  dict(params=model.head.parameters(), lr=args.head_learning_rate)], weight_decay=.01)
    settings = dict(objective="document_classifier", readout=readout, frozen_embeddings=True, require_perfect_validation=args.require_perfect_validation,
                    encoder_learning_rate=args.encoder_learning_rate, head_learning_rate=args.head_learning_rate, selection_patience=args.selection_patience, batch_size=8, threshold=.5, eval_every=args.eval_every,
                    checkpoint_selection="generated_family_quality", sampling="paired_queries_uniform_source" if args.paired_queries else "uniform_units", loss="unweighted_bce")
    args.run_dir.mkdir(parents=True)
    info = dict(data=str(args.data), dataset_fingerprint=digest, source=str(args.init_from),
                source_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(), device=args.device,
                train_units=len(train), validation_units=len(val), settings=settings, optimizer_reset=True, test_examples_used=0,
                note="Predição semântica por unidade; JSON/referências por construção. Validação de desenvolvimento, sem teste cego; sem integração na API.")
    (args.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
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
            indices = (paired_indices(train, paired_pools, 8, generator) if args.paired_queries
                       else torch.randint(len(train), (8,), generator=generator).tolist())
            items = [train[i] for i in indices]
            ids, lengths = unit_batch(items, args.device, backbone.config.block_size)
            labels = torch.tensor([item["labels"] for item in items], device=args.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ids, lengths), labels)
            if not torch.isfinite(loss): raise ValueError("Loss não finita; checkpoint anterior preservado.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            checkpoint(step, "last.pt")
            result = score(model, val, validation, args.device); key = quality_key(result)
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
