"""Dados de treino de checklist a partir de artigos CC-BY anotados por duas IAs independentes.

Só entram itens (documento, critério) em que as duas anotações têm o mesmo status. Por item:
- atende/contradiz: cada trecho marcado pelas duas vira `atendido`/`nao_atendido`; o candidato BM25
  mais bem colocado que nenhuma marcou vira `evidencia_insuficiente` (mesmo requisito, rótulo oposto);
- não encontrado: os 2 melhores candidatos BM25 viram `evidencia_insuficiente` (citam o tema sem cumprir).
Splits por documento (--validation lista documentos de validação); nada vai para teste.
--balance iguala atendido e insuficiente por critério; critérios sem positivo saem.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions

STATUS = {"atende": "atendido", "contradiz": "nao_atendido"}


def row(key, document, split, req, text, status, rights):
    units = [dict(id="page:1", text=text)]
    evidence = [] if status == "evidencia_insuficiente" else [dict(status=status, unit="page:1")]
    return dict(id=key, project_id=document, source_document=document, split=split, task="grounded_" + status,
                grounding_schema=2, evaluation_profile="checklist_" + status, requirement=req, units=units,
                document=document + ".txt", prompt=prompt_for(req, units, 2),
                target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
                provenance=dict(kind="ai_annotated_requirement", license=rights.split(" ")[0], rights=rights,
                                agreement="two_independent_ai_annotations", rules="ai/annotation/criterios-v1.md",
                                paragraph_sha256=hashlib.sha256(text.encode()).hexdigest(), independent_human_review=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--first", type=Path, nargs="+", required=True, help="Exports da anotação A.")
    p.add_argument("--second", type=Path, nargs="+", required=True, help="Exports da anotação B.")
    p.add_argument("--validation", nargs="*", default=[])
    p.add_argument("--balance", action="store_true", help="Equilibra rótulos por critério (remove o atalho da consulta).")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Saída deve ser nova.")
    tasks = json.loads(a.tasks.read_text())
    folder = {d["id"]: Path(d["folder"]) for d in tasks["documents"]}
    items = {(i["document"], i["criterion"]): i for i in tasks["items"]}
    load = lambda paths: {(i["document"], i["criterion"]): i for f in paths for i in json.loads(f.read_text())["items"]}
    first, second = load(a.first), load(a.second)
    shared = sorted(set(first) & set(second))
    agree = [k for k in shared if first[k]["status"] == second[k]["status"]]
    rows, skipped = [], Counter()
    for k in agree:
        document, criterion = k; task = items[k]
        rights = json.loads((folder[document] / "source.json").read_text())["rights"]
        if not rights.startswith("CC-BY"): skipped["license"] += 1; continue
        split = "validation" if document in a.validation else "train"
        req, status = task["requirement"], first[k]["status"]
        marked = lambda x: {(p["unit"], p["start"]) for p in x["passages"]}
        both, any_ = marked(first[k]) & marked(second[k]), marked(first[k]) | marked(second[k])
        candidates = {(c["unit"], c["start"]): c for c in task["passages"]}
        base = f"{document}-{criterion}"
        if status in STATUS:
            if not both: skipped["no_shared_passage"] += 1; continue
            for n, key in enumerate(sorted(both)):
                rows.append(row(f"{base}-pos{n}", document, split, req, candidates[key]["text"], STATUS[status], rights))
            rest = [c for c in task["passages"] if (c["unit"], c["start"]) not in any_]
            if rest: rows.append(row(f"{base}-neg", document, split, req, rest[0]["text"], "evidencia_insuficiente", rights))
        else:
            for n, c in enumerate(task["passages"][:2]):
                rows.append(row(f"{base}-neg{n}", document, split, req, c["text"], "evidencia_insuficiente", rights))
    if a.balance:
        # Mesmo número de atendido e insuficiente por critério e split: a consulta sozinha não decide.
        import random
        groups = {}
        for r in rows: groups.setdefault((r["split"], r["id"].split("-")[-2]), []).append(r)
        rows = []
        for key, group in sorted(groups.items()):
            pos = [r for r in group if r["task"] != "grounded_evidencia_insuficiente"]
            neg = [r for r in group if r["task"] == "grounded_evidencia_insuficiente"]
            random.Random(42).shuffle(pos); random.Random(42).shuffle(neg)
            n = min(len(pos), len(neg)); skipped["balance_dropped"] += len(group) - 2 * n
            rows += pos[:n] + neg[:n]
    seen, unique = set(), []
    for r in rows:
        if r["prompt"].strip().casefold() in seen: skipped["duplicate"] += 1; continue
        seen.add(r["prompt"].strip().casefold()); unique.append(r)
    a.output.mkdir(parents=True)
    path = a.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique))
    load_instructions(path)
    observed = sum(first[k]["status"] == second[k]["status"] for k in shared) / len(shared)
    fa, fb = Counter(first[k]["status"] for k in shared), Counter(second[k]["status"] for k in shared)
    expected = sum(fa[s] * fb[s] for s in fa) / len(shared) ** 2
    manifest = dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), items=len(shared), agreed=len(agree),
                    agreement=round(observed, 3), kappa=round((observed - expected) / (1 - expected), 3),
                    agreed_status=dict(Counter(first[k]["status"] for k in agree)),
                    counts=dict(Counter(f"{r['split']}:{r['evaluation_profile']}" for r in unique)),
                    validation_documents=a.validation, skipped=dict(skipped),
                    limitations=["Rótulos de IA sem revisão humana.", "Artigos científicos, não TCCs.",
                                 "Trecho não marcado pode sustentar o critério com outras palavras."])
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__": main()
