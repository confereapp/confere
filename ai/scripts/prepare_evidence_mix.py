"""Mistura para classificador sobre base nova: controle CC amostrado, relações públicas DF e matriz acadêmica DG.

Validação = validação DG (51 pares acadêmicos + 410 casos antigos), igual a DG/DH/DI/DJ.
Ficam fora: reserva DG (test) e validação/teste DF, preservados para comparação posterior.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from confere_ai.instruct import load_instructions


def rows_of(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--relations", type=Path, nargs="+", required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--control-rows", type=int, default=15000)
    p.add_argument("--relation-rows", type=int, help="Limite por arquivo de relações (amostra com seed); padrão: todas.")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Saída deve ser nova.")
    matrix = rows_of(a.matrix)
    taken = {r["id"] for r in matrix} | {r["prompt"].strip().casefold() for r in matrix}
    control = [r for r in rows_of(a.control) if r["split"] == "train"
               and r["id"] not in taken and r["prompt"].strip().casefold() not in taken]
    random.Random(42).shuffle(control)
    relations = []
    for path in a.relations:
        rows = [r for r in rows_of(path) if r["split"] == "train"
                and r["id"] not in taken and r["prompt"].strip().casefold() not in taken]
        random.Random(42).shuffle(rows); relations += rows[:a.relation_rows]
    rows = [r for r in matrix if r["split"] in {"train", "validation"}] + control[:a.control_rows] + relations
    # DF já reaproveita casos do controle: mantém a primeira ocorrência de cada ID/prompt.
    seen, unique = set(), []
    for r in rows:
        keys = {r["id"], r["prompt"].strip().casefold()}
        if not keys & seen: unique.append(r); seen |= keys
    dropped, rows = len(rows) - len(unique), unique
    a.output.mkdir(parents=True)
    target = a.output / "examples.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    load_instructions(target)  # mesmas validações de split, duplicatas, alvos e partição usadas no treino
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in [("control", a.control), ("matrix", a.matrix)] + [(f"relations:{x.parent.name}", x) for x in a.relations]}
    manifest = dict(dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), sources=sources,
                    control_rows=min(a.control_rows, len(control)), relation_rows=len(relations), duplicates_dropped=dropped, seed=42,
                    counts=dict(Counter(f"{r['split']}:{r.get('evaluation_profile', r['task'])}" for r in rows)),
                    excluded="Reserva DG (test) e validação/teste DF; nenhuma linha de teste incluída.")
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(rows=len(rows), duplicates_dropped=dropped, splits=Counter(r["split"] for r in rows)), ensure_ascii=False))


if __name__ == "__main__": main()
