"""Prepara tarefas de anotação: para cada TCC e critério, os trechos mais parecidos por BM25.

O anotador lê só os candidatos (e a página, se precisar) em vez do TCC inteiro. Os trechos preservam
página e posição no texto extraído, para virarem citações conferíveis na importação.
"""
import argparse
import hashlib
import json
from pathlib import Path

from confere_ai.evidence_classifier import sentence_parts
from confere_ai.extract import extract_document
from confere_ai.project_review import bm25_rank


def passages(document, size=600):
    """Junta frases consecutivas de cada página até ~`size` caracteres, guardando as posições."""
    result = []
    for unit in document["units"]:
        start = 0; current = ""
        for sentence in sentence_parts(unit["text"]):
            if current and len(current) + len(sentence) > size:
                result.append(dict(unit=unit["id"], start=start, end=start + len(current), text=current))
                start += len(current); current = ""
            current += sentence
        if current.strip():
            result.append(dict(unit=unit["id"], start=start, end=start + len(current), text=current))
    return [p for p in result if p["text"].strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", type=Path, nargs="+", required=True, help="Pastas maua-* com ocr-v1/searchable.pdf.")
    p.add_argument("--checklist", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Saída deve ser nova.")
    criteria = [line.strip() for line in a.checklist.read_text().splitlines() if line.strip()]
    documents, items = [], []
    for folder in a.documents:
        pdf = folder / "ocr-v1" / "searchable.pdf"
        document = extract_document(pdf)
        source = json.loads((folder / "source.json").read_text())
        chunks = passages(document)
        documents.append(dict(id=folder.name, title=source["title"], url=source["url"], sha256=document["sha256"],
                              pages=len(document["units"]), folder=str(folder)))
        for number, criterion in enumerate(criteria, 1):
            ranked = bm25_rank(criterion, [c["text"] for c in chunks])[:a.top_k]
            items.append(dict(document=folder.name, criterion=f"C{number}", requirement=criterion,
                              passages=[dict(rank=r, **chunks[i]) for r, i in enumerate(ranked, 1)]))
    a.output.write_text(json.dumps(dict(schema_version=1, checklist_sha256=hashlib.sha256(a.checklist.read_bytes()).hexdigest(),
                                        criteria=criteria, documents=documents, items=items), ensure_ascii=False, indent=1))
    print(json.dumps(dict(documents=len(documents), items=len(items)), ensure_ascii=False))


if __name__ == "__main__": main()
