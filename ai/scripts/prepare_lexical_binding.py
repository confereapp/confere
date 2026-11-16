"""Treina associação literal requisito/evidência com vocabulário aberto autoral."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import torch
from confere_ai.grounding import prompt_for, judge
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from expand_context_answers import names

FORMS = [
    ("O projeto inclui {}.", "O projeto não inclui {}."),
    ("O documento registra presença de {}.", "O documento registra ausência de {}."),
    ("A equipe confirmou a existência de {}.", "A equipe confirmou a inexistência de {}."),
    ("O relatório comprova a presença de {}.", "O relatório comprova a ausência de {}."),
    ("Há evidência da implementação de {}.", "Não há suporte implementado para {}."),
]
WORDS = ["testes", "backup", "cronograma", "cadastro", "login", "documentação", "bibliografia", "índice", "resumo",
         "objetivos", "legendas", "referências", "segurança", "acessibilidade", "assinaturas", "metodologia", "anexos"]


def expected(units, feature):
    evidence = []
    for i, unit in enumerate(units, 1):
        for status in sorted({status for topic, status in unit["facts"] if topic == feature}):
            evidence.append(dict(status=status, unit=f"page:{i}"))
    statuses = {row["status"] for row in evidence}
    status = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
    return dict(status=status, evidence=evidence)


def documents(count=2000, feature_prefix="recurso "):
    rng = random.Random(151130)
    rows = []
    for i in range(count):
        invented = [feature_prefix + value.casefold() for value in names(rng)]
        if i % 3 == 0: feature, other = invented
        elif i % 3 == 1: feature, other = invented[0], rng.choice(WORDS)
        else: feature, other = rng.sample(WORDS, 2)
        pos, neg = rng.choice(FORMS)
        def clause(topic, status):
            return dict(text=(pos if status == "atendido" else neg).format(topic), facts=[(topic, status)])
        positive, negative = clause(feature, "atendido"), clause(feature, "nao_atendido")
        other_pos, other_neg = clause(other, "atendido"), clause(other, "nao_atendido")
        mode = i % 9
        if mode == 0: units = [other_pos]
        elif mode == 1: units = [positive, other_neg]
        elif mode == 2: units = [negative, other_pos]
        elif mode == 3: units = [positive, copy.deepcopy(positive), other_neg]
        elif mode == 4: units = [negative, copy.deepcopy(negative), other_pos]
        elif mode == 5: units = [positive, negative, other_pos]
        elif mode == 6: units = [dict(text=positive["text"]+" "+negative["text"], facts=positive["facts"]+negative["facts"]), other_pos]
        elif mode == 7: units = [dict(text=positive["text"]+" "+other_neg["text"], facts=positive["facts"]+other_neg["facts"])]
        else: units = [dict(text=f"A reunião discutiu {feature}; não há comprovação de implementação nessa ata.", facts=[]), other_neg]
        units.append(dict(text=f"Registro de equipe {rng.randrange(1000000,9999999)}.", facts=[]))
        rng.shuffle(units)
        assert len(units) <= 4
        aliases = [dict(id=f"page:{j+1}", text=unit["text"]) for j, unit in enumerate(units)]
        for j, topic in enumerate([feature, other]):
            target = expected(units, topic)
            group = f"binding-train-{i}-{j}"
            requirement = f"O projeto deve incluir {topic}."
            row = dict(id=group, project_id=group, split="train", task="grounded_"+target["status"], grounding_schema=2,
                       document=f"binding-{i}.txt", requirement=requirement, units=copy.deepcopy(aliases), prompt=prompt_for(requirement, aliases, 2),
                       target=json.dumps(target, ensure_ascii=False), evaluation_profile="lexical_binding", scenario=dict(units=units, feature=topic),
                       provenance=dict(kind="synthetic_authored", note="Estados e referências por construção; inclui nomes de recursos inventados e contrastes de requisito no mesmo texto. Sem revisão humana independente."))
            assert judge(row["target"], row)["exact_match"]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source)
    added = documents()
    rows = source + added
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), additional_documents=len(added),
                    splits=dict(Counter(r["split"] for r in rows)),
                    note="Validação C preservada literalmente; apenas treino adicional de associação entre requisito e evidência. Sem consulta a teste reservado, desenvolvimento não cego.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
