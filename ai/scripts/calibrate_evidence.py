"""Calibra limiares em desenvolvimento; não altera pesos ou alvos."""
import argparse
from io import BytesIO
import hashlib
import itertools
import json
from pathlib import Path
import torch
from confere_ai.evidence_classifier import EvidencePredictor, examples, unit_batch, response
from confere_ai.generative_selection import quality_key
from confere_ai.grounding import judge
from confere_ai.instruct import load_instructions, acceptance


def evaluate_thresholds(rows, probabilities, thresholds):
    samples = []; cursor = 0
    for row in rows:
        count = len(row["units"])
        actual = response(row["units"], probabilities[cursor:cursor+count], thresholds); cursor += count
        samples.append(dict(id=row["id"], task=row["task"], prompt=row["prompt"], expected=row["target"], actual=actual,
                            evaluation_profile=row.get("evaluation_profile", "default"), **judge(actual, row)))
    assert cursor == len(probabilities)
    return dict(samples=samples, correct=sum(r["exact_match"] for r in samples), total=len(samples))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "data", "run-dir"]: p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps"], default="mps"); args = p.parse_args()
    if args.run_dir.exists(): p.error("Saída já existe.")
    payload = args.checkpoint.read_bytes(); checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    predictor = EvidencePredictor(checkpoint, args.device); torch.set_num_threads(4)
    rows, digest = load_instructions(args.data); rows = [r for r in rows if r["split"] == "validation"]
    if not rows or any(not r["task"].startswith("grounded_") for r in rows): p.error("Exige validação documental.")
    encoded = examples(rows, predictor.tokenizer, predictor.config.block_size, readout=predictor.readout); probabilities = []
    with torch.no_grad():
        for offset in range(0, len(encoded), 8):
            ids, lengths = unit_batch(encoded[offset:offset+8], args.device, predictor.config.block_size)
            probabilities.extend(predictor.model(ids, lengths).sigmoid().cpu().tolist())
    grid = []; best = selected = thresholds = None
    for pair in itertools.product([.5, .6, .7, .8, .9, .95, .99], repeat=2):
        result = evaluate_thresholds(rows, probabilities, pair); key = quality_key(result)
        grid.append(dict(thresholds=pair, quality=key))
        if best is None or key > best: best, selected, thresholds = key, result, pair
    args.run_dir.mkdir(parents=True); (args.run_dir/"source.pt").write_bytes(payload)
    checkpoint["settings"] = dict(checkpoint["settings"], thresholds=list(thresholds), calibration="development_grid_v1")
    torch.save(checkpoint, args.run_dir/"best.pt")
    info = dict(data=str(args.data), dataset_fingerprint=digest, source=str(args.checkpoint),
                source_sha256=hashlib.sha256(payload).hexdigest(), device=args.device, settings=checkpoint["settings"],
                test_examples_used=0, training_steps=0, note="Limiares selecionados em desenvolvimento. Nenhum peso alterado. Novo teste independente obrigatório; sem promoção automática.")
    (args.run_dir/"run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (args.run_dir/"grid.json").write_text(json.dumps(grid, indent=2))
    evaluation = dict(baseline=evaluate_thresholds(rows, probabilities, predictor.thresholds), selected=selected,
                      selected_step=checkpoint["step"], quality=best, acceptance=acceptance(selected), test_examples_used=0, adopted=False)
    (args.run_dir/"evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(json.dumps(dict(step=checkpoint["step"], thresholds=thresholds, quality=best, correct=selected["correct"], total=selected["total"])), flush=True)


if __name__ == "__main__": main()
