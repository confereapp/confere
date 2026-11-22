"""Restaura continuações autorais e contrasta intenções em contextos variados."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_informal_dialogues import TARGETS


def build(source, authored, count=2400):
    rows = list(source)
    reserved = {r["prompt"].strip().casefold() for r in source if r["split"] != "train"}
    contexts = {json.dumps([r.get("history", []), r["prompt"].strip().casefold()], ensure_ascii=False):
                (r["split"], r["target"]) for r in source}
    concepts = {}
    for row in authored:
        if row["split"] == "train" and row["task"] == "conversation_definition" and row.get("turn_index") == 1:
            previous, definition = row["history"][-1]
            if previous.startswith("O que é "):
                concepts[previous[len("O que é "):-1]] = (definition, row["target"])
    if len(concepts) != 4:
        raise ValueError("Quatro conceitos autorais completos são obrigatórios.")
    seeds = [[], [["oi", TARGETS["hello"]]], [["qual seu nome?", TARGETS["identity"]]]]
    seeds += [r["history"][-2:] for r in source if r["split"] == "train" and r["task"].startswith("conversation_")
              and not r["task"].startswith("conversation_context_answer_") and r.get("history")]
    banks = {
        "hello": ["opa, bora bater papo", "e aí, podemos conversar um pouco?", "olá, vamos conversar hoje?", "fala aí, Confere!"],
        "identity": ["você pode dizer seu nome?", "qual nome identifica você?", "lembra como você se chama?"],
        "function": ["você pode explicar sua função?", "qual apoio você oferece nos projetos?", "fale sobre o que você faz"],
        "scope": ["você trabalha com quais tipos de projeto?", "qual categoria de projeto você analisa?", "que projetos fazem parte do seu escopo?"],
        "praise": ["obrigado por essa explicação", "curti, mandou bem", "valeu mesmo pela ajuda, Confere", "boa, gostei bastante"],
        "negation": ["por ora não tenho projeto algum", "nenhum projeto para mostrar hoje", "não há projeto comigo no momento"],
    }
    definitions = ["Você pode definir {}?", "Você sabe explicar a palavra {}?", "Qual é a definição de {}?",
                   "Explique para mim {} em uma frase.", "Diga o que significa {} para um projeto.", "Na sua explicação, o que é {}?"]
    examples = ["mostre um exemplo disso", "pode exemplificar esse conceito?", "quero um exemplo do que explicou", "dê um exemplo"]
    rng = random.Random(161104)
    for i in range(count):
        history = [list(p) for p in rng.choice(seeds)]
        kind = i % 8
        if kind == 0:
            term = rng.choice(sorted(concepts))
            definition, example = concepts[term]
            turns = [("definition", rng.choice(definitions).format(term), definition),
                     ("definition", rng.choice(examples), example)]
        elif kind == 1:
            turns = [("guidance", rng.choice(["não escolhi ainda o que pedir para você", "não consegui decidir como quero sua ajuda", "estou em dúvida sobre o que pedir"]),
                      "Sem problema. Podemos falar de objetivos, requisitos ou evidências."),
                     ("guidance", rng.choice(["essas sugestões ainda me deixam indeciso", "ainda não decidi entre essas opções", "sigo sem saber qual dessas opções escolher"]),
                      "Vamos começar pelo tema do seu projeto. Qual é o tema?")]
        else:
            task = list(banks)[kind - 2]
            turns = [("slang_" + task, rng.choice(banks[task]), TARGETS[task])]
        proposed = []
        for index, (task, prompt, target) in enumerate(turns):
            # Continuações autorais existentes podem coincidir; novas raízes não copiam validação.
            if index == 0 and prompt.strip().casefold() in reserved:
                raise ValueError("Nova raiz coincide com formulação reservada.")
            key = json.dumps([history, prompt.strip().casefold()], ensure_ascii=False)
            if key in contexts and contexts[key] != ("train", target):
                break
            group = f"intent-replay-{i}"
            row = dict(id=f"{group}-{index}", project_id=group, split="train", task="conversation_" + task,
                       history=[list(p) for p in history], turn_index=index, response_tokens=256, prompt=prompt, target=target,
                       provenance=dict(kind="synthetic_authored", note="Continuações e alvos autorais; variantes de intenção somente no treino, sem revisão humana independente."))
            proposed.append((key, row)); history.append([prompt, target])
        else:
            for key, row in proposed:
                contexts[key] = ("train", row["target"]); rows.append(row)
    assert [r for r in rows if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source)
    authored_path = Path("ai/data/authored/conversations.jsonl")
    authored = [json.loads(line) for line in authored_path.read_text().splitlines() if line.strip()]
    rows = build(source, authored)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    load_instructions(path)
    manifest = dict(source_sha256=digest, authored_sha256=hashlib.sha256(authored_path.read_bytes()).hexdigest(),
                    dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), splits=dict(Counter(r["split"] for r in rows)),
                    added=len(rows)-len(source), note="Somente treino alterado. Validação de desenvolvimento preservada; nenhum teste reservado usado para ajuste.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
