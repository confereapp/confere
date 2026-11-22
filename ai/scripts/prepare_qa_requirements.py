"""Requisitos no formato de checklist derivados de perguntas FaQuAD, sobre parágrafos íntegros.

Cada pergunta vira um requisito ("O texto deve informar ..."). O parágrafo que contém a resposta é
`atendido`; um parágrafo do mesmo documento sem a resposta, escolhido por BM25 entre os mais parecidos,
é `evidencia_insuficiente`. Cada requisito aparece com os dois rótulos: a consulta sozinha não decide.
Não há `nao_atendido`: perguntas extrativas não trazem contradições.
Splits por documento, iguais aos da matriz DG (validação e reserva continuam fora do treino).
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.project_review import bm25_rank
from confere_ai.qa_source import SOURCE, REVISION, TRAIN_SHA256
from confere_ai.relation_partition import PROTOCOL, components

TEMPLATES = ("O texto deve informar {q}.", "O documento deve responder: {q}?", "O trecho deve indicar {q}.")


def requirement(question, index):
    q = question.strip().rstrip("?").strip()
    q = q[0].lower() + q[1:]
    return TEMPLATES[index % len(TEMPLATES)].format(q=q)


def row(key, title, split, req, text, status, question, answer):
    units = [dict(id="page:1", text=text)]
    evidence = [] if status == "evidencia_insuficiente" else [dict(status=status, unit="page:1")]
    return dict(id=key, project_id="faquad-" + title, source_document="faquad:" + title, split=split,
                task="grounded_" + status, grounding_schema=2, evaluation_profile="qa_requirement_" + status,
                requirement=req, units=units, document=f"faquad-{title}.txt", prompt=prompt_for(req, units, 2),
                target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
                provenance=dict(kind="public_qa_requirement", source=SOURCE, revision=REVISION, train_sha256=TRAIN_SHA256,
                                license="CC-BY-4.0", partition_protocol=PROTOCOL, question=question, answer=answer,
                                paragraph_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                derivation="question_template_v1", independent_human_review=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="train.json do snapshot FaQuAD auditado.")
    p.add_argument("--matrix-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Saída deve ser nova.")
    if hashlib.sha256(a.source.read_bytes()).hexdigest() != TRAIN_SHA256: p.error("Snapshot FaQuAD alterado.")
    documents = json.loads(a.matrix_manifest.read_text())["documents"]
    split_of = {d.removeprefix("faquad:"): s for s, names in documents.items() for d in names}
    paragraphs = {}
    for document in json.loads(a.source.read_text())["data"]:
        paragraphs.setdefault(document["title"], []).extend(document["paragraphs"])
    rows, skipped, index = [], Counter(), 0
    for title, items in paragraphs.items():
        split = split_of.get(title, "train")
        texts = [item["context"] for item in items]
        for position, item in enumerate(items):
            for qa in item["qas"]:
                question, answer = qa["question"], qa["answers"][0]["text"]
                req = requirement(question, index); index += 1
                key = f"faquad-req-{title}-{position}-{qa['id']}"
                rows.append(row(key, title, split, req, item["context"], "atendido", question, answer))
                others = [i for i in bm25_rank(req, texts) if i != position and answer not in texts[i]]
                if not others: skipped["no_negative"] += 1; continue
                rows.append(row(key + "-neg", title, split, req, texts[others[0]], "evidencia_insuficiente", question, answer))
    seen, unique = set(), []
    for r in rows:
        k = r["prompt"].strip().casefold()
        if k in seen: skipped["duplicate"] += 1; continue
        seen.add(k); unique.append(r)
    # Frases quase iguais entre documentos unem componentes; o componente vai para o split mais protegido.
    keys, priority, assigned = components(unique), {"train": 0, "validation": 1, "test": 2}, {}
    for r in unique:
        k = keys[r["id"]]
        assigned[k] = max(assigned.get(k, "train"), r["split"], key=priority.get)
    moved = Counter()
    for r in unique:
        k = keys[r["id"]]
        if r["split"] != assigned[k]: moved[f"{r['split']}->{assigned[k]}"] += 1
        r["split"], r["project_id"] = assigned[k], k
    skipped.update(moved)
    a.output.mkdir(parents=True)
    path = a.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique))
    load_instructions(path)
    manifest = dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_sha256=TRAIN_SHA256,
                    counts=dict(Counter(f"{r['split']}:{r['evaluation_profile']}" for r in unique)),
                    documents={s: sorted({r["source_document"] for r in unique if r["split"] == s}) for s in ("train", "validation", "test")},
                    skipped=dict(skipped), templates=TEMPLATES,
                    limitations=["Requisitos por modelo fixo de frase, sem revisão humana.",
                                 "Negativo pode responder a pergunta com outras palavras.", "Sem nao_atendido."])
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(rows=len(unique), counts=manifest["counts"], skipped=manifest["skipped"]), ensure_ascii=False))


if __name__ == "__main__": main()
