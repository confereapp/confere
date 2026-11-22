"""Baixa artigos SciELO em português com licença CC-BY para virar dado de treino de requisitos.

Para cada ISSN, lista artigos pela API ArticleMeta, mantém os de idioma original português,
confirma a licença CC-BY na página do artigo e baixa o PDF (camada de texto original, sem OCR).
Cada artigo vira uma pasta `scielo-<pid>/` no formato das pastas maua-* (original.pdf,
ocr-v1/searchable.pdf, source.json), para reaproveitar prepare_annotation.py.
"""
import argparse
import hashlib
import html
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

from confere_ai.extract import extract_document

API = "http://articlemeta.scielo.org/api/v1"
UA = {"User-Agent": "confere-research/1.0"}


def get(url, binary=False):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        return (data, response.geturl()) if binary else (data.decode("utf-8", "replace"), response.geturl())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--issn", nargs="+", required=True)
    p.add_argument("--per-journal", type=int, default=4)
    p.add_argument("--since", default="2018-01-01")
    p.add_argument("--min-pages", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    taken = []
    for issn in a.issn:
        listing = json.loads(get(f"{API}/article/identifiers/?collection=scl&issn={issn}&from={a.since}&limit=200")[0])
        got = 0
        for item in listing["objects"]:
            if got >= a.per_journal: break
            pid = item["code"]; folder = a.output / f"scielo-{pid}"
            if folder.exists(): got += 1; continue
            try: meta = json.loads(get(f"{API}/article/?collection=scl&code={pid}")[0])
            except Exception as error: print(json.dumps(dict(pid=pid, skipped=str(error)))); continue
            article = meta["article"]
            if article.get("v40", [{}])[0].get("_") != "pt" or meta.get("document_type") != "research-article": continue
            try: page, url = get(f"https://www.scielo.br/scielo.php?script=sci_arttext&pid={pid}&tlng=pt")
            except Exception as error: print(json.dumps(dict(pid=pid, skipped=str(error)))); continue
            license = re.search(r"creativecommons\.org/licenses/(by(?:-sa)?)/([\d.]+)", page)
            pdf = re.search(r'href="([^"]*format=pdf[^"]*lang=pt[^"]*)"', page)
            if not license or not pdf: continue
            try: data, _ = get("https://www.scielo.br" + html.unescape(pdf.group(1)), binary=True)
            except Exception as error: print(json.dumps(dict(pid=pid, skipped=str(error)))); continue
            if not data.startswith(b"%PDF"): continue
            folder.mkdir(); (folder / "ocr-v1").mkdir()
            (folder / "original.pdf").write_bytes(data)
            shutil.copy(folder / "original.pdf", folder / "ocr-v1" / "searchable.pdf")
            document = extract_document(folder / "ocr-v1" / "searchable.pdf")
            text = sum(len(u["text"].strip()) for u in document["units"])
            words = re.findall(r"\w+", " ".join(u["text"] for u in document["units"]).lower())
            english = sum(w in {"the", "of", "and", "that", "is", "with"} for w in words) > sum(w in {"de", "que", "não", "uma", "para"} for w in words)
            if len(document["units"]) < a.min_pages or text < 15000 or english:  # metadado "pt" às vezes aponta PDF em inglês
                shutil.rmtree(folder); continue
            title = next((t["_"] for t in article.get("v12", []) if t.get("l") == "pt"), article.get("v12", [{}])[0].get("_", ""))
            (folder / "source.json").write_text(json.dumps(dict(
                schema_version=1, title=title, year=meta.get("publication_year"), url=url, doi=meta.get("doi"), issn=issn,
                original_file="original.pdf", sha256=hashlib.sha256(data).hexdigest(), ocr_file="ocr-v1/searchable.pdf",
                review="Camada de texto do PDF original; sem OCR (ocr-v1 é cópia, mantida pelo formato das pastas).",
                rights=f"CC-{license.group(1).upper()} {license.group(2)} (https://creativecommons.org/licenses/{license.group(1)}/{license.group(2)}/), "
                       "confirmada na página do artigo; atribuição: autores e periódico conforme DOI.",
                training_eligible=True, split="train"), ensure_ascii=False, indent=2))
            taken.append(dict(pid=pid, issn=issn, pages=len(document["units"]), chars=text, title=title[:80]))
            got += 1; time.sleep(1)
            print(json.dumps(taken[-1], ensure_ascii=False), flush=True)
    print(json.dumps(dict(articles=len(taken)), ensure_ascii=False))


if __name__ == "__main__": main()
