"""Casos autorais com evidências múltiplas; famílias separadas por split."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from prepare_document_contrasts import FORMS, TOPICS


def documents(count=1200):
    rows = []
    for split, size in (("train", count), ("validation", 24)):
        rng = random.Random(711 if split == "train" else 712)
        for i in range(size):
            feature = rng.choice(TOPICS[split])
            forms = FORMS[split] if split != "train" else FORMS[split] + [
                ("Foi registrada a presença de {}.", "Foi registrada a ausência de {}."),
                ("Está disponível: {}.", "Está ausente: {}."),
            ]
            positive, negative = rng.choice(forms)
            mode = i % 6
            labels = ([], ["atendido"], ["nao_atendido"],
                      ["atendido", "atendido"], ["nao_atendido", "nao_atendido"],
                      ["atendido", "nao_atendido"])[mode]
            parts = [("Registro da equipe.", None), (f"Identificador do projeto: {split}-{i}.", None)]
            for index, status in enumerate(labels):
                pos, neg = rng.choice(forms) if index and split == "train" else (positive, negative)
                parts.append(((pos if status == "atendido" else neg).format(feature), status))
            rng.shuffle(parts)
            units = [dict(id=f"page:{j+1}", text=text) for j, (text, _) in enumerate(parts)]
            evidence = [dict(status=status, unit=units[j]["id"]) for j, (_, status) in enumerate(parts) if status]
            status = labels[0] if len(set(labels)) == 1 else "evidencia_insuficiente"
            requirement = f"O projeto deve incluir {feature}."
            rows.append(dict(id=f"multi-{split}-{i}", project_id=f"multi-{split}-{i}", split=split,
                             task="grounded_"+status, grounding_schema=2, requirement=requirement,
                             units=units, document=f"fiction-{split}-{i}.txt", mode=mode,
                             prompt=prompt_for(requirement, units, 2),
                             target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
                             provenance=dict(kind="synthetic_authored", note="Rótulo por construção; sem revisão independente.")))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = documents()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    _, fingerprint = load_instructions(path)
    manifest = dict(dataset_sha256=fingerprint, counts=dict(Counter(r["split"] for r in rows)),
                    modes=dict(Counter(str(r["mode"]) for r in rows)), test_examples_used=0,
                    note="Piloto de contrato; validação separa vocabulário e frases. Não mede documentos reais ou conversa.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
