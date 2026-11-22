"""Contrastes de relação sem atalho da consulta, derivados das linhas InferBR do corpus DF.

Para cada par (premissa P, hipótese H, rótulo L), no mesmo split do DF:
- original: H com P -> L;
- negação: "Não é verdade que h" com P -> L invertido (implicação <-> contradição; neutra continua insuficiente);
- premissa não relacionada: H com P' de outro componente do mesmo split, escolhida entre vizinhos BM25
  (posições 3-14, evitando paráfrases) -> evidencia_insuficiente; idem para a negação.
Cada consulta aparece com dois rótulos e cada premissa com vários: nem consulta nem trecho sozinhos decidem.
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.nli_source import STATES
from confere_ai.project_review import bm25_rank
from confere_ai.relation_partition import verify_partition, words

FLIP = {"atendido": "nao_atendido", "nao_atendido": "atendido", "evidencia_insuficiente": "evidencia_insuficiente"}
LABEL = {status: label for label, status in STATES.items()}


def negate(claim):
    claim = claim.strip().rstrip(".")
    return "Não é verdade que " + claim[0].lower() + claim[1:] + "."


def derived(row, suffix, requirement, unit_text, status, kind):
    units = [dict(id="page:1", text=unit_text)]
    evidence = [] if status == "evidencia_insuficiente" else [dict(status=status, unit="page:1")]
    provenance = dict(row["provenance"], source_label=LABEL[status], derivation=kind,
                      original_label=row["provenance"]["source_label"])
    return dict(row, id=row["id"] + "-" + suffix, task="grounded_" + status, requirement=requirement, units=units,
                prompt=prompt_for(requirement, units, 2), evaluation_profile="contrast_nli_" + status,
                target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False), provenance=provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Saída deve ser nova.")
    rows = [json.loads(l) for l in a.relations.read_text().split("\n") if l.strip()]
    nli = [r for r in rows if r["provenance"]["kind"] == "public_reviewed_nli"]
    rng = random.Random(42)
    result = []
    for split in ("train", "validation", "test"):
        group = [r for r in nli if r["split"] == split]
        premises = sorted({r["units"][0]["text"] for r in group})
        owner = defaultdict(set)
        for r in group: owner[r["units"][0]["text"]].add(r["project_id"])
        for r in group:
            status = json.loads(r["target"])["status"]
            claim, premise = r["requirement"], r["units"][0]["text"]
            ranked = [premises[i] for i in bm25_rank(claim, premises)
                      if r["project_id"] not in owner[premises[i]]][3:15]
            other = rng.choice(ranked)
            result += [dict(r, evaluation_profile="contrast_nli_" + status),
                       derived(r, "neg", negate(claim), premise, FLIP[status], "negated_claim"),
                       derived(r, "unrel", claim, other, "evidencia_insuficiente", "unrelated_premise_bm25"),
                       derived(r, "neg-unrel", negate(claim), other, "evidencia_insuficiente", "negated_unrelated_premise_bm25")]
    seen, unique = set(), []
    for r in result:
        key = r["prompt"].strip().casefold()
        if key not in seen: seen.add(key); unique.append(r)
    # Uma negação curta pode ficar quase igual (Jaccard >= .8) a frase de outro split; esses derivados saem.
    index = defaultdict(lambda: defaultdict(set))  # token -> split -> frases
    for r in unique:
        for text in [r["requirement"]] + [u["text"] for u in r["units"]]:
            tokens = frozenset(words(text))
            for t in tokens: index[t][r["split"]].add(tokens)
    def leaks(r):
        for text in [r["requirement"]] + [u["text"] for u in r["units"]]:
            tokens = frozenset(words(text))
            for t in tokens:
                for split, others in index[t].items():
                    if split != r["split"] and any(len(tokens & o) / len(tokens | o) >= .8 for o in others): return True
        return False
    kept = [r for r in unique if "derivation" not in r["provenance"] or not leaks(r)]
    leaking, unique = len(unique) - len(kept), kept
    verify_partition(unique)
    a.output.mkdir(parents=True)
    path = a.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique))
    load_instructions(path)
    queries = defaultdict(Counter)
    for r in unique: queries[(r["split"], r["requirement"])][r["task"]] += 1
    ceiling = {s: sum(max(c.values()) for (split, _), c in queries.items() if split == s) for s in ("train", "validation", "test")}
    manifest = dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    relations_sha256=hashlib.sha256(a.relations.read_bytes()).hexdigest(),
                    counts=dict(Counter(f"{r['split']}:{r['evaluation_profile']}" for r in unique)),
                    duplicates_removed=len(result) - len(unique) - leaking, cross_split_derived_removed=leaking,
                    query_only_ceiling=ceiling,
                    split_policy="Mesmo split e componente do DF; premissas não relacionadas vêm do mesmo split.",
                    limitations=["Negação por modelo fixo de frase.",
                                 "Premissa não relacionada pode, raramente, sustentar a hipótese; sem revisão humana local."])
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(rows=len(unique), splits=Counter(r["split"] for r in unique), query_only_ceiling=ceiling), ensure_ascii=False))


if __name__ == "__main__": main()
