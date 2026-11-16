"""Alvos locais anotados para os lotes efetivamente usados na revisão textual."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.project_review import batches_for
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_document_contrasts import FORMS, TOPICS
from prepare_document_lexical import LEXICAL

PROFILES = ("scope_short", "scope_long", "tail_conflict", "middle_evidence", "short_conflict", "mention_long")
BACKGROUND = ("A equipe registrou reuniões e distribuiu as tarefas da semana.",
              "Os integrantes organizaram as atividades e anotaram os responsáveis.",
              "O documento descreve o calendário das reuniões e a divisão do trabalho.")


def scene_for(split, family, profile, rng, *, forms=None):
    feature, other = rng.sample(TOPICS[split], 2)
    forms = forms if forms is not None else (LEXICAL if split == "train" else FORMS[split])
    positive, negative = rng.choice(forms)
    first = rng.choice(("atendido", "nao_atendido"))
    second = "nao_atendido" if first == "atendido" else "atendido"
    def clause(topic, status):
        return ((positive if status == "atendido" else negative).format(topic), topic, status)
    def noise(low, high):
        return (" ".join(rng.choice(BACKGROUND) for _ in range(rng.randint(low, high))), None, None)
    a, b = clause(feature, first), clause(other, second)
    opposite = clause(feature, second)
    if profile == "scope_short":
        segments = [a, b]
    elif profile == "scope_long":
        segments = [a, noise(30, 50), b, noise(15, 25)]
    elif profile == "tail_conflict":
        segments = [a, noise(60, 90), opposite, b]
    elif profile == "middle_evidence":
        segments = [noise(15, 30), a, noise(20, 35), b, noise(10, 20)]
    elif profile == "short_conflict":
        segments = [a, opposite, b]
    else:
        segments = [(f"A equipe discutiu {feature}; a ata não comprova sua implementação.", None, None),
                    noise(35, 60), b, noise(10, 20)]
    segments.insert(0, (f"🙂 Registro de projeto {rng.randrange(10000000, 99999999)}.", None, None))
    text = ""; annotations = []
    for sentence, topic, status in segments:
        if text:
            text += " "
        start = len(text); text += sentence
        if topic:
            annotations.append(dict(feature=topic, status=status, start=start, end=len(text), quote=sentence))
    if len(text) > 8000:
        raise ValueError("Fixture excede o limite textual da API.")
    name = f"api-{split}-{family}-{profile}"
    document = dict(name=name+".txt", sha256=hashlib.sha256(text.encode()).hexdigest(), format="text",
                    warnings=[], units=[dict(id="text:1", text=text)])
    expected = []
    for topic in (feature, other):
        statuses = {a["status"] for a in annotations if a["feature"] == topic}
        expected.append(dict(status=next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente",
                             conflicting_evidence=len(statuses) > 1))
    return dict(id=name, split=split, profile=profile, document=document, annotations=annotations,
                features=[feature, other], requirements=[f"O projeto deve incluir {t}." for t in (feature, other)], expected=expected)


def local_rows(scene, tokenizer, context_size=1024, response_tokens=256):
    planned = [batches_for([scene["document"]], r, tokenizer, context_size, 2, response_tokens)
               for r in scene["requirements"]]
    # Um fragmento de frase não recebe um rótulo semântico inventado.
    for batches in planned:
        for parts in batches:
            for part in parts:
                for annotation in scene["annotations"]:
                    overlap = part["char_start"] < annotation["end"] and annotation["start"] < part["char_end"]
                    contains = part["char_start"] <= annotation["start"] and annotation["end"] <= part["char_end"]
                    if overlap and not contains:
                        return None
    rows = []
    for req_index, (requirement, batches) in enumerate(zip(scene["requirements"], planned)):
        for batch_index, parts in enumerate(batches):
            units = [dict(id=f"page:{i+1}", text=p["text"]) for i, p in enumerate(parts)]
            evidence = []
            for unit, part in zip(units, parts):
                statuses = {a["status"] for a in scene["annotations"] if a["feature"] == scene["features"][req_index]
                            and part["char_start"] <= a["start"] and a["end"] <= part["char_end"]}
                evidence.extend(dict(status=status, unit=unit["id"]) for status in sorted(statuses))
            statuses = {e["status"] for e in evidence}
            status = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
            category = "conflict" if len(statuses) > 1 else status
            rows.append(dict(id=f"{scene['id']}-r{req_index}-b{batch_index}", project_id=scene["id"],
                             split=scene["split"], task="grounded_"+status, grounding_schema=2,
                             evaluation_profile="api_batch_"+category, training_profile=scene["profile"],
                             requirement=requirement, units=units, document=scene["document"]["name"],
                             source_sha256=scene["document"]["sha256"],
                             sources=[{k: v for k, v in p.items() if k != "text"} for p in parts],
                             prompt=prompt_for(requirement, units, 2),
                             target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
                             provenance=dict(kind="synthetic_authored", note="Alvo local por intervalos autorais; loteador real da API. Sem revisão independente.")))
    return rows


def build(tokenizer, train_families=160, validation_families=3):
    rows = []; validation_cases = []; skipped = Counter()
    for split, count in (("train", train_families), ("validation", validation_families)):
        rng = random.Random(1041 if split == "train" else 1042)
        for family in range(count):
            for profile in PROFILES:
                scene = scene_for(split, family, profile, rng)
                local = local_rows(scene, tokenizer)
                if local is None:
                    skipped[f"{split}:split_evidence_sentence"] += 1
                    continue
                rows.extend(local)
                if split == "validation":
                    validation_cases.append(scene)
    return rows, validation_cases, dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Saída deve ser nova.")
    source, source_hash = load_instructions(args.source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"]["block_size"] != 1024 or checkpoint["settings"].get("grounding_schema") != 2:
        parser.error("Currículo exige contrato V2 e contexto 1024.")
    tokenizer = ByteBPE(checkpoint["merges"])
    generated, cases, skipped = build(tokenizer)
    # Replay curto estratificado; os exemplos retirados continuam no corpus anterior.
    rng = random.Random(1043); pools = defaultdict(list); retained = []
    for row in source:
        if row["split"] == "train" and row["id"].startswith("intensive-"):
            pools[row["evaluation_profile"]].append(row)
        else:
            retained.append(row)
    for pool in pools.values():
        rng.shuffle(pool); retained.extend(pool[:200])
    informative = [r for r in generated if r["split"] == "train" and json.loads(r["target"])["evidence"]]
    absent = [r for r in generated if r["split"] == "train" and not json.loads(r["target"])["evidence"]]
    rng.shuffle(absent)
    selected_ids = {r["id"] for r in informative + absent[:len(informative)]}
    candidates = retained + [r for r in generated if r["split"] != "train" or r["id"] in selected_ids]
    rows = []; seen = {}
    for row in candidates:
        if "history" not in row:
            key = row["prompt"].strip().casefold()
            if key in seen:
                previous = seen[key]
                if previous["split"] != row["split"] or json.loads(previous["target"]) != json.loads(row["target"]):
                    raise ValueError("Contexto repetido com split ou alvo diferente.")
                continue
            seen[key] = row
        encode_sample(row, tokenizer, 1024)
        if len(tokenizer.encode(row["target"]))+1 > 256:
            raise ValueError("Alvo excede orçamento de resposta.")
        rows.append(row)
    previous_validation = [r for r in source if r["split"] != "train"]
    assert [r for r in rows if r["split"] != "train"][:len(previous_validation)] == previous_validation
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    _, fingerprint = load_instructions(path)
    (args.output / "validation-projects.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    manifest = dict(dataset_sha256=fingerprint, source_sha256=source_hash,
                    counts=dict(Counter(r["split"] for r in rows)), legacy_validation_preserved=len(previous_validation),
                    local_validation_rows=sum(r["split"] == "validation" and r["id"].startswith("api-") for r in rows),
                    full_validation_projects=len(cases), generated_rows=len(generated), skipped=skipped,
                    profiles=dict(Counter(r.get("evaluation_profile", "conversation") for r in rows)),
                    test_examples_used=0, note="Lotes reais, rótulos autorais por intervalos, validação de desenvolvimento. Nenhum teste reservado ou documento real.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
