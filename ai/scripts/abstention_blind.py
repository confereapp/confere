"""Abstenção no teste cego: o modelo decide só quando a confiança é alta; o resto vira "revisar".

Passo 1 (--checkpoint): grava as probabilidades por trecho (BM25 top-k, mesmo fluxo da API).
Passo 2 (--probs): varre limiares e escolhe-os por deixar-um-documento-de-fora (mira precisão >= --target
nos outros 7 TCCs e aplica no oitavo), para que o número reportado não use o próprio documento.
Escore do item = fração dos k trechos com P(atendido) >= 0,5 (a maioria atual é escore > 0,5).
"""
import argparse
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if not n: return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 3), round((c + m) / d, 3)]


def record(checkpoint, candidates, top_k, device):
    from confere_ai.evidence_classifier import EvidencePredictor
    from confere_ai.extract import extract_document
    from confere_ai.project_review import review_documents
    original, seen = EvidencePredictor.probabilities, []
    def spy(self, requirement, units):
        values = original(self, requirement, units); seen.append(values[0]); return values
    EvidencePredictor.probabilities = spy
    items = []
    for path in candidates:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        names = [n for n in rows[0]["documents"] if n != rows[0]["requirement"]["source"]["document"]]
        documents = [extract_document(path.parent / n) for n in names]
        result = review_documents(documents, [r["requirement"]["text"] for r in rows], checkpoint, device, top_k)
        for row, req in zip(rows, result["requirements"]):
            probs, seen[:] = seen[:len(req["calls"])], seen[len(req["calls"]):]
            items.append(dict(id=row["id"], document=path.parent.name, expected=row["expected"]["status"],
                              actual=req["status"], probs=probs))
        assert not seen
        print(path.parent.name, flush=True)
    return items


def score(item): return sum(p[0] >= .5 for p in item["probs"]) / len(item["probs"])


def decide(item, hi, lo):
    s = score(item)
    return "atendido" if s >= hi else "evidencia_insuficiente" if s <= lo else "revisar"


def metrics(items, hi, lo):
    dec = [(decide(i, hi, lo), i["expected"]) for i in items]
    approved = [e for d, e in dec if d == "atendido"]; decided = [(d, e) for d, e in dec if d != "revisar"]
    ok = sum(e == "atendido" for e in approved)
    right = sum(d == e or (d == "evidencia_insuficiente" and e != "atendido") for d, e in decided)
    return dict(hi=hi, lo=lo, approved=len(approved), approved_correct=ok,
                precision=round(ok / len(approved), 3) if approved else None, ci95=wilson(ok, len(approved)),
                recall=round(ok / sum(i["expected"] == "atendido" for i in items), 3),
                coverage=round(len(decided) / len(items), 3),
                decided_accuracy=round(right / len(decided), 3) if decided else None)


GRID = [round(x / 8, 3) for x in range(9)]


def choose(items, target):
    # Maior cobertura com precisão >= alvo e ao menos 5 aprovações; lo simétrico ao que mantém acerto alto.
    best = None
    for hi in GRID[1:]:
        for lo in [g for g in GRID if g < hi]:
            m = metrics(items, hi, lo)
            if m["approved"] < 5 or m["precision"] < target or (m["decided_accuracy"] or 0) < target: continue
            if best is None or m["coverage"] > best["coverage"]: best = m
    return best


def analyse(items, target):
    curve = [metrics(items, hi, hi - 1e-9 if hi > 0 else -1) for hi in GRID[1:]]  # sem abstenção: lo = hi
    rows, held = [], []
    for doc in sorted({i["document"] for i in items}):
        pick = choose([i for i in items if i["document"] != doc], target)
        test = [i for i in items if i["document"] == doc]
        if pick is None: held += [("revisar", i["expected"]) for i in test]; rows.append(dict(document=doc, chosen=None)); continue
        held += [(decide(i, pick["hi"], pick["lo"]), i["expected"]) for i in test]
        rows.append(dict(document=doc, hi=pick["hi"], lo=pick["lo"]))
    approved = [e for d, e in held if d == "atendido"]; decided = [(d, e) for d, e in held if d != "revisar"]
    ok = sum(e == "atendido" for e in approved)
    right = sum(d == e or (d == "evidencia_insuficiente" and e != "atendido") for d, e in decided)
    lodo = dict(target=target, approved=len(approved), approved_correct=ok,
                precision=round(ok / len(approved), 3) if approved else None, ci95=wilson(ok, len(approved)),
                coverage=round(len(decided) / len(held), 3), decided=len(decided),
                decided_accuracy=round(right / len(decided), 3) if decided else None,
                decided_accuracy_ci95=wilson(right, len(decided)), thresholds=rows)
    return dict(items=len(items), no_abstention_curve=curve, leave_one_document_out=lodo,
                in_sample_best=choose(items, target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--candidates", type=Path, nargs="*")
    p.add_argument("--probs", type=Path, required=True, help="Arquivo de probabilidades (gerado se --checkpoint).")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default="mps")
    p.add_argument("--target", type=float, default=.9)
    a = p.parse_args()
    if a.checkpoint:
        if a.probs.exists(): p.error("Arquivo de probabilidades já existe.")
        a.probs.write_text(json.dumps(record(a.checkpoint, a.candidates, a.top_k, a.device)))
    items = json.loads(a.probs.read_text())
    print(json.dumps(analyse(items, a.target), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    demo = [dict(document="d", expected="atendido", probs=[[.9, 0]] * 8), dict(document="d", expected="nao_encontrado", probs=[[.1, 0]] * 8)]
    assert decide(demo[0], .75, .25) == "atendido" and decide(demo[1], .75, .25) == "evidencia_insuficiente"
    assert metrics(demo, .75, .25)["precision"] == 1.0
    main()
