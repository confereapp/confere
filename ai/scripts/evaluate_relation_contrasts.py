"""Avalia um classificador no teste de contrastes e no teste DF, contra atalhos de consulta e de trecho."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from confere_ai.evidence_classifier import EvidencePredictor
from evaluate_relation_pilot import shortcut_baseline
from train_evidence_classifier import score


def rows_of(path):
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def by_class(rows, correct):
    result = defaultdict(lambda: [0, 0])
    for row, ok in zip(rows, correct):
        status = json.loads(row["target"])["status"]; result[status][0] += ok; result[status][1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in sorted(result.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--contrasts", type=Path, required=True)
    p.add_argument("--relations", type=Path, required=True)
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    a = p.parse_args()
    predictor = EvidencePredictor(torch.load(a.checkpoint, map_location="cpu", weights_only=True), a.device)
    report = {}
    for name, path in [("contrasts", a.contrasts), ("df", a.relations)]:
        rows = rows_of(path)
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test" and r["provenance"]["kind"] == "public_reviewed_nli"]
        gold = [json.loads(r["target"])["status"] for r in test]
        model = [s["exact_match"] for s in score(predictor.model, [], test, a.device, predictor.thresholds,
                                                  predict_units=predictor.predict_units)["samples"]]
        entry = dict(total=len(test), model=sum(model), model_by_class=by_class(test, model),
                     majority=max(Counter(gold).values()))
        for side in ("query", "body"):
            hits = [x == y for x, y in zip(shortcut_baseline(train, test, side), gold)]
            entry[side + "_only"] = sum(hits)
        report[name] = entry
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
