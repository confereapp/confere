"""Amplia somente treino com pares contrafactuais; preserva a validação anterior."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_document_contrasts import TOPICS
from prepare_document_lexical import LEXICAL, balanced_replay
from prepare_evidence_robustness import PROFILES


def intensive_rows(scenarios=1600):
    rng = random.Random(911)
    rows = []
    topics = TOPICS["train"] + ["controle de acesso", "proteção de dados", "documentação", "acessibilidade"]
    for scenario in range(scenarios):
        feature = rng.choice(topics)
        other = rng.choice([topic for topic in topics if topic != feature])
        positive, negative = rng.choice(LEXICAL)
        positive2, negative2 = rng.choice(LEXICAL)
        identifier = rng.randrange(100000, 999999)
        for mode, profile in enumerate(PROFILES):
            labels = (["atendido"], ["nao_atendido"], [], ["atendido", "atendido"],
                      ["nao_atendido", "nao_atendido"], ["atendido", "nao_atendido"],
                      ["atendido", "nao_atendido"], ["atendido"])[mode]
            parts = []
            for index, status in enumerate(labels):
                pos, neg = (positive, negative) if not index else (positive2, negative2)
                parts.append(((pos if status == "atendido" else neg).format(feature), {status}))
            if mode == 6:
                if rng.randrange(2):
                    parts.reverse()
                parts = [(" ".join(p[0] for p in parts), set(labels))]
            distractors = [(f"A equipe discutiu {feature}; a ata não comprova sua implementação.", set()),
                           (rng.choice((positive, negative)).format(other), set())]
            parts += distractors if mode == 7 else distractors[:rng.randint(0 if parts else 1, 2)]
            # Preservar de uma a quatro unidades, inclusive a entrada textual única da API.
            if mode in (0, 1, 2, 6, 7) and rng.randrange(2):
                parts = [(" ".join(p[0] for p in parts), set().union(*(p[1] for p in parts)))]
            background = " A equipe registrou tarefas, reuniões e responsáveis na documentação."
            parts = [(text + background*rng.choice((0, 0, 1, 2, 3)), statuses) for text, statuses in parts]
            rng.shuffle(parts)
            prefix = rng.choice(("", "🙂 ", "Análise: ", f"Registro {identifier}. "))
            parts[0] = (prefix + parts[0][0], parts[0][1])
            units = [dict(id=f"page:{j+1}", text=text) for j, (text, _) in enumerate(parts)]
            evidence = [dict(status=status, unit=units[j]["id"]) for j, (_, statuses) in enumerate(parts)
                        for status in sorted(statuses)]
            status = labels[0] if len(set(labels)) == 1 else "evidencia_insuficiente"
            requirement = f"O projeto deve incluir {feature}."
            rows.append(dict(id=f"intensive-{scenario}-{mode}", project_id=f"intensive-{scenario}", split="train",
                             task="grounded_"+status, grounding_schema=2, evaluation_profile=profile,
                             requirement=requirement, units=units, document=f"intensive-{scenario}.txt",
                             prompt=prompt_for(requirement, units, 2),
                             target=json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False),
                             provenance=dict(kind="synthetic_authored", note="Par contrafactual autoral; rótulo por construção. Sem revisão independente.")))
    # Variações curtas podem coincidir; manter uma única amostra por contexto.
    unique = {}
    for row in rows:
        unique.setdefault(row["prompt"].strip().casefold(), row)
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Saída deve ser nova.")
    source, source_hash = load_instructions(args.source)
    replay_source, replay_hash = load_instructions(args.replay_source)
    base = [r for r in source if r["split"] != "train" or r["task"].startswith("grounded_")]
    replay = balanced_replay(replay_source, per_family=160)
    rows = base + intensive_rows() + replay
    seen = set()
    unique = []
    for row in rows:
        if "history" not in row:
            key = row["prompt"].strip().casefold()
            if key in seen:
                continue
            seen.add(key)
        unique.append(row)
    assert [r for r in unique if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    context_size = checkpoint["config"]["block_size"]
    max_sequence = 0
    for row in unique:
        ids, _ = encode_sample(row, tokenizer, context_size)
        max_sequence = max(max_sequence, len(ids))
        if len(tokenizer.encode(row["target"]))+1 > 256:
            raise ValueError(f"{row['id']}: resposta excede 256 tokens.")
        if row["task"].startswith("grounded_") and len(tokenizer.encode(row["prompt"]))+2+256 > context_size:
            raise ValueError(f"{row['id']}: falta espaço para a resposta; geração recusada sem truncamento.")
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in unique))
    _, fingerprint = load_instructions(path)
    manifest = dict(dataset_sha256=fingerprint, source_sha256=source_hash, replay_source_sha256=replay_hash,
                    counts=dict(Counter(r["split"] for r in unique)), replay_train=len(replay),
                    document_profiles=dict(Counter(r.get("evaluation_profile", "conversation") for r in unique if r["split"] == "train")),
                    validation_unchanged=True, test_examples_used=0,
                    tokenizer_checkpoint=str(args.checkpoint), context_size=context_size,
                    response_tokens=256, max_sequence_tokens=max_sequence,
                    note="Ampliação exclusivamente autoral do treino; validação de desenvolvimento preservada, sem teste reservado.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
