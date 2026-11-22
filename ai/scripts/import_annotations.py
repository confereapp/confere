"""Converte anotações exportadas por ai/annotation/annotate.html no contrato de avaliação de documentos reais.

Gera `examples.blind.jsonl` em cada pasta maua-*, legível por evaluate_real_documents.py. Com --second,
mede concordância entre anotadores (bruta e kappa de Cohen) antes de qualquer uso.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from confere_ai.extract import extract_document

STATUS = {"atende": "atendido", "contradiz": "nao_atendido", "nao_encontrado": "evidencia_insuficiente"}


def kappa(first, second):
    shared = sorted(set(first) & set(second))
    if not shared: return None, 0
    observed = sum(first[k] == second[k] for k in shared) / len(shared)
    a, b = Counter(first[k] for k in shared), Counter(second[k] for k in shared)
    expected = sum(a[s] * b[s] for s in a) / len(shared) ** 2
    return dict(items=len(shared), agreement=round(observed, 3),
                kappa=round((observed - expected) / (1 - expected), 3) if expected < 1 else None), len(shared)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--second", type=Path, help="Anotações independentes de outra pessoa, para concordância.")
    a = p.parse_args()
    tasks = json.loads(a.tasks.read_text()); data = json.loads(a.annotations.read_text())
    if data["checklist_sha256"] != tasks["checklist_sha256"]: p.error("Anotações de outro checklist.")
    if a.second:
        other = json.loads(a.second.read_text())
        labels = lambda d: {(i["document"], i["criterion"]): i["status"] for i in d["items"]}
        print(json.dumps(dict(concordancia=kappa(labels(data), labels(other))[0]), ensure_ascii=False))
    folders = {d["id"]: Path(d["folder"]) for d in tasks["documents"]}
    checklist = "\n".join(tasks["criteria"]) + "\n"
    by_document, problems = {}, []
    for item in data["items"]:
        folder = folders[item["document"]]
        document = extract_document(folder / "ocr-v1" / "searchable.pdf")
        units = {u["id"]: u["text"] for u in document["units"]}
        evidence = [dict(document="ocr-v1/searchable.pdf", unit=x["unit"], quote=units[x["unit"]][x["start"]:x["end"]].strip())
                    for x in item["passages"]]
        if item["extra_quote"].strip():
            unit = f"page:{item['extra_page'].strip()}"
            quote = " ".join(item["extra_quote"].split())
            if unit not in units or quote not in " ".join(units[unit].split()):
                problems.append(f"{item['document']} {item['criterion']}: citação extra não encontrada em {unit}")
            else: evidence.append(dict(document="ocr-v1/searchable.pdf", unit=unit, quote=quote))
        status = STATUS[item["status"]]
        if status != "evidencia_insuficiente" and not evidence:
            problems.append(f"{item['document']} {item['criterion']}: decisão sem evidência"); continue
        number = item["criterion"]
        by_document.setdefault(item["document"], []).append(dict(
            schema_version=1, id=f"{item['document']}-{number}", project_id=item["document"], split="test",
            provenance=dict(kind="real", usage_rights="Somente avaliação; licença de treinamento não confirmada.",
                            reviewed_by=data["annotator"], annotation_tool="ai/annotation/annotate.html",
                            exported=data["exported"], derived_from="original.pdf", note=item["note"]),
            documents=["checklist-v1.txt", "ocr-v1/searchable.pdf"],
            requirement=dict(id=number, text=item["requirement"],
                             source=dict(document="checklist-v1.txt", unit="page:1", quote=item["requirement"])),
            expected=dict(status=status, evidence=[] if status == "evidencia_insuficiente" else evidence,
                          justification=item["note"] or "Anotação humana na ferramenta de candidatos BM25.")))
    for name, rows in by_document.items():
        folder = folders[name]
        (folder / "checklist-v1.txt").write_text(checklist)
        (folder / "examples.blind.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts = Counter(r["expected"]["status"] for rows in by_document.values() for r in rows)
    print(json.dumps(dict(documentos=len(by_document), criterios=sum(map(len, by_document.values())),
                          rotulos=counts, problemas=problems), ensure_ascii=False, indent=1))


if __name__ == "__main__": main()
