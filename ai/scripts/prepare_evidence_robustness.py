"""Corpus autoral de evidências difíceis com replay de conversas completas."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from prepare_document_contrasts import FORMS, TOPICS
from prepare_multiple_evidence import documents

PROFILES = ("single_positive", "single_negative", "mention_only", "multiple_positive",
            "multiple_negative", "split_conflict", "shared_conflict", "other_topic")


def robustness_rows(count=960):
    rows = []
    for split, size in (("train", count), ("validation", 96)):
        rng = random.Random(811 if split == "train" else 812)
        for i in range(size):
            feature = rng.choice(TOPICS[split])
            other = rng.choice([t for t in TOPICS[split] if t != feature])
            forms = FORMS[split] + ([("Foi registrada a presença de {}.", "Foi registrada a ausência de {}.")]
                                    if split == "train" else [])
            mode = i % len(PROFILES)
            labels = (["atendido"], ["nao_atendido"], [], ["atendido", "atendido"],
                      ["nao_atendido", "nao_atendido"], ["atendido", "nao_atendido"],
                      ["atendido", "nao_atendido"], ["atendido"])[mode]
            parts = []
            for status in labels:
                positive, negative = rng.choice(forms)
                parts.append(((positive if status == "atendido" else negative).format(feature), {status}))
            if mode == 6:
                parts = [(" ".join(p[0] for p in parts), set(labels))]
            distractors = [
                (f"A equipe discutiu {feature} na reunião. Isso não informa sua implementação.", set()),
                (f"O projeto não inclui {other}.", set()),
            ]
            if mode != 7:
                distractors = distractors[:rng.randint(1, 2)]
            parts += distractors
            background = " A equipe registrou as atividades e distribuiu tarefas."
            parts = [(text + background*rng.choice((0, 2, 5)), statuses) for text, statuses in parts]
            rng.shuffle(parts)
            # A API recebe uma unidade textual; cobrir também o caso sem separação em páginas.
            if mode in (0, 1, 2, 6, 7) and i % 3 == 0:
                parts = [("\n".join(p[0] for p in parts), set().union(*(p[1] for p in parts)))]
            parts[0] = (f"Registro {split}-{i}. " + parts[0][0], parts[0][1])
            units = [dict(id=f"page:{j+1}", text=text) for j, (text, _) in enumerate(parts)]
            evidence = [dict(status=status, unit=units[j]["id"]) for j, (_, statuses) in enumerate(parts)
                        for status in sorted(statuses)]
            consolidated = labels[0] if len(set(labels)) == 1 else "evidencia_insuficiente"
            requirement = f"O projeto deve incluir {feature}."
            rows.append(dict(id=f"robust-{split}-{i}", project_id=f"robust-{split}-{i}", split=split,
                             task="grounded_"+consolidated, grounding_schema=2,
                             evaluation_profile=PROFILES[mode], requirement=requirement, units=units,
                             document=f"robust-{split}-{i}.txt", prompt=prompt_for(requirement, units, 2),
                             target=json.dumps(dict(status=consolidated, evidence=evidence), ensure_ascii=False),
                             provenance=dict(kind="synthetic_authored", note="Rótulos por construção; distratores e conflitos autorais, sem revisão independente.")))
    return rows


def conversation_replay(rows, train_groups=8, validation_groups=1):
    groups = defaultdict(list)
    for row in rows:
        if row["task"].startswith("conversation_") and row["split"] in {"train", "validation"}:
            groups[(row["split"], row["project_id"])].append(row)
    counts = Counter()
    selected = []
    for (split, _), turns in sorted(groups.items()):
        task = turns[0]["task"]
        limit = train_groups if split == "train" else validation_groups
        if counts[(split, task)] < limit:
            selected.extend(turns)
            counts[(split, task)] += 1
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Saída deve ser nova.")
    source, source_hash = load_instructions(args.replay_source)
    base = [dict(row, evaluation_profile="previous_multiple_evidence") for row in documents()]
    replay = conversation_replay(source)
    rows = base + robustness_rows() + replay
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    _, fingerprint = load_instructions(path)
    manifest = dict(dataset_sha256=fingerprint, replay_source_sha256=source_hash,
                    counts=dict(Counter(r["split"] for r in rows)),
                    profiles=dict(Counter(r.get("evaluation_profile", "conversation") for r in rows)),
                    test_examples_used=0,
                    note="Validação autoral e conhecida; replay por conversas completas. Sem teste reservado ou documentos reais.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
