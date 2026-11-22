"""Pontua avaliações do teste cego (evaluate_real_documents.py) contra os rótulos importados.

Precisão de "atendido" com intervalo de Wilson 95%, evidência correta nas aprovações (sobreposição
com algum trecho anotado) e controles sem modelo: sempre aprovar, só a consulta (maioria do critério
nos outros TCCs) e BM25 sozinho (limiar no melhor trecho, escolhido nos outros TCCs).
--alternate troca os rótulos de alguns documentos por outro arquivo exportado (ex.: outro anotador).
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from confere_ai.extract import extract_document
from confere_ai.project_review import bm25_rank, terms

STATUS = {"atende": "atendido", "contradiz": "nao_atendido", "nao_encontrado": "evidencia_insuficiente"}


def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def norm(s): return " ".join(s.split())


def overlaps(quote, gold):
    q, g = norm(quote), norm(gold)
    if q in g or g in q: return True
    a, b = set(terms(q)), set(terms(g))
    return bool(a) and len(a & b) / len(a) >= .6


def precision(pairs):
    approved = [e for p, e in pairs if p == "atendido"]
    k = sum(e == "atendido" for e in approved)
    return dict(approved=len(approved), correct=k, precision=round(k / len(approved), 3) if approved else None,
                ci95=wilson(k, len(approved)), recall=round(k / max(1, sum(e == "atendido" for _, e in pairs)), 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, nargs="+", required=True, help="examples.blind.jsonl")
    p.add_argument("--runs", type=Path, nargs="*", default=[])
    p.add_argument("--alternate", type=Path, help="Export de anotação cujos documentos substituem os rótulos.")
    p.add_argument("--tasks", type=Path, help="Tarefas de anotação (texto dos trechos), exigido com --alternate.")
    a = p.parse_args()
    gold = {}
    for path in a.labels:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            gold[r["id"]] = dict(project=r["project_id"], criterion=r["requirement"]["id"], text=r["requirement"]["text"],
                                 status=r["expected"]["status"], quotes=[e["quote"] for e in r["expected"]["evidence"]],
                                 folder=path.parent)
    if a.alternate:
        text = {(i["document"], i["criterion"], x["unit"], x["start"]): x["text"]
                for i in json.loads(a.tasks.read_text())["items"] for x in i["passages"]}
        for item in json.loads(a.alternate.read_text())["items"]:
            g = gold[f"{item['document']}-{item['criterion']}"]; g["status"] = STATUS[item["status"]]
            g["quotes"] = [text[item["document"], item["criterion"], x["unit"], x["start"]] for x in item["passages"]]
            g["quotes"] += [item["extra_quote"]] if item["extra_quote"].strip() else []
    report = dict(labels=dict(Counter(g["status"] for g in gold.values())), items=len(gold))
    ids = sorted(gold)
    report["always_approve"] = precision([("atendido", gold[i]["status"]) for i in ids])
    by_crit = {}
    for i in ids: by_crit.setdefault(gold[i]["criterion"], []).append(i)
    query = [("atendido" if 2 * sum(gold[j]["status"] == "atendido" for j in by_crit[gold[i]["criterion"]] if gold[j]["project"] != gold[i]["project"])
              > len(by_crit[gold[i]["criterion"]]) - 1 else "evidencia_insuficiente", gold[i]["status"]) for i in ids]
    report["query_only"] = precision(query)
    # BM25 sozinho: melhor score do critério no documento (trechos de ~600 caracteres das tarefas).
    from prepare_annotation import passages
    chunks, top = {}, {}
    for i in ids:
        g = gold[i]
        if g["project"] not in chunks: chunks[g["project"]] = [c["text"] for c in passages(extract_document(g["folder"] / "ocr-v1" / "searchable.pdf"))]
        texts = chunks[g["project"]]; q = set(terms(g["text"]))
        top[i] = max(len(q & set(terms(t))) / len(q) for t in texts)  # fração de termos da consulta no melhor trecho
    bm25 = []
    for i in ids:
        train = [j for j in ids if gold[j]["project"] != gold[i]["project"]]
        best = max(sorted({top[j] for j in train}), key=lambda t: sum((top[j] >= t) == (gold[j]["status"] == "atendido") for j in train))
        bm25.append(("atendido" if top[i] >= best else "evidencia_insuficiente", gold[i]["status"]))
    report["bm25_only"] = precision(bm25)
    for run in a.runs:
        pairs, ok_evidence, rows = [], 0, []
        for f in sorted(run.glob("maua-*.json")):
            data = json.loads(f.read_text())
            for j, req in zip(data["judgments"], data["trace"]["requirements"]):
                g = gold[j["id"]]; pairs.append((j["actual"], g["status"]))
                if j["actual"] == "atendido" and g["status"] == "atendido":
                    quotes = [e["quote"] for e in req["evidence"] if e.get("model_status", "atendido") == "atendido"]
                    golds = g["quotes"]
                    hit = any(overlaps(q, x) for q in quotes for x in golds)
                    ok_evidence += hit; rows.append(dict(id=j["id"], evidence_hit=hit))
        r = precision(pairs)
        r["evidence_correct"] = ok_evidence; r["evidence_rate"] = round(ok_evidence / r["correct"], 3) if r["correct"] else None
        r["status_accuracy"] = round(sum(x == y for x, y in pairs) / len(pairs), 3); r["items"] = len(pairs)
        r["actual"] = dict(Counter(x for x, _ in pairs))
        report[run.name] = r
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__": main()
