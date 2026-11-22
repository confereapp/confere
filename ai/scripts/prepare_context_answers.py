"""Currículo autoral de respostas curtas verificáveis pelo contexto, sem fatos externos."""
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
from prepare_document_lexical import balanced_replay

ABSENT = "Não tenho informação suficiente para responder."
CONFLICT = "Há informações conflitantes no texto."
NAMES = ["Ana", "Bruno", "Carla", "Daniel", "Elisa", "Fabio", "Gabriel", "Helena", "Igor", "Julia", "Lucas", "Maria", "Nina", "Paula", "Rafael", "Sofia"]
FIELDS = {
    "nome": ("O projeto se chama {}.", ["Qual é o nome do projeto?", "Como se chama o projeto?", "Diga o nome do projeto."], ["Que nome foi dado ao projeto?", "Qual nome aparece no texto?"]),
    "responsavel": ("A pessoa responsável é {}.", ["Quem é a pessoa responsável?", "Quem é responsável pelo projeto?", "Diga quem é responsável."], ["A quem foi atribuída a responsabilidade?", "Qual pessoa responde pelo projeto?"]),
    "prazo": ("O prazo informado é de {} dias.", ["Qual é o prazo?", "Em quantos dias é o prazo?", "Diga o prazo informado."], ["Quantos dias foram informados?", "Qual duração foi indicada para o prazo?"]),
    "linguagem": ("A linguagem usada é {}.", ["Qual linguagem é usada?", "Diga a linguagem do projeto.", "Qual é a linguagem informada?"], ["Que linguagem aparece no contexto?", "Qual linguagem foi escolhida?"]),
}


def answer(facts, field):
    values = facts.get(field, [])
    if not values: return ABSENT
    if len(set(values)) > 1: return CONFLICT
    value = values[0]
    if field == "prazo": return str(value) + " dias."
    if field == "testes": return "Sim." if value == "presente" else "Não."
    return str(value) + "."


def build(count=1200, validation_per_family=12):
    rng = random.Random(151127)
    rows = []
    for split, size in [("train", count), ("validation", validation_per_family)]:
        for family in ["fact", "missing", "conflict", "support", "memory"]:
            for i in range(size):
                group = f"context-answer-{split}-{family}-{i}"
                names = NAMES if split == "train" else ["Clara", "Davi", "Renata", "Victor"]
                person, other = rng.sample(names, 2)
                project = rng.choice(["Atlas", "Aurora", "Ponte", "Farol", "Jardim", "Nuvem"])
                facts = dict(nome=[project], responsavel=[person], prazo=[rng.randint(1, 90)], linguagem=[rng.choice(["Python", "Java", "C", "Ruby"])])
                field = rng.choice(list(FIELDS))
                history = []
                def add(prompt, target, index=0):
                    rows.append(dict(id=f"{group}-{index}", project_id=group, split=split,
                                     task="conversation_context_answer_"+family, prompt=prompt, target=target,
                                     history=[list(p) for p in history], turn_index=index, response_tokens=256,
                                     scenario=dict(facts=facts, field=field),
                                     provenance=dict(kind="synthetic_authored", note="Alvo derivado de fatos autorais; formas de validação e nomes separados. Sem revisão humana independente.")))
                    history.append([prompt, target])
                if family == "memory":
                    # A resposta final depende da correção, não do primeiro nome.
                    for j, (prompt, target) in enumerate([
                        (f"Meu nome é {person}." if split == "train" else f"Pode me chamar de {person}.", f"Prazer, {person}."),
                        (f"Corrigindo: meu nome é {other}." if split == "train" else f"Na verdade, me chamo {other}.", f"Prazer, {other}."),
                        ("Qual é meu nome agora?" if split == "train" else "Depois da correção, como eu me chamo?", f"Seu nome é {other}.")]):
                        add(prompt, target, j)
                    continue
                if family == "missing": del facts[field]
                if family == "conflict":
                    alternate = dict(nome="Caminho", responsavel=other, prazo=100, linguagem="Lua")[field]
                    facts[field].append(alternate)
                if family == "support":
                    field = "testes"
                    # Três condições igualmente frequentes, intercaladas no mesmo formato de pergunta.
                    state = i % 3
                    facts = {"testes": ["presente" if state == 0 else "ausente"]} if state < 2 else {}
                    sentences = ["A equipe está registrada.", "O projeto inclui documentação."]
                    if state < 2: sentences.append("O projeto inclui testes." if state == 0 else "O projeto não inclui testes.")
                    question = rng.choice(["O projeto inclui testes?", "Há testes no projeto?"] if split == "train" else ["O texto confirma a presença de testes?", "Os testes fazem parte do projeto?"])
                else:
                    sentences = [FIELDS[key][0].format(value) for key, values in facts.items() for value in values]
                    question = rng.choice(FIELDS[field][1 if split == "train" else 2])
                rng.shuffle(sentences)
                # Identificador sem correlação com o alvo mantém contextos distintos, inclusive contrafactuais.
                prompt = f"Registro {hashlib.sha256(group.encode()).hexdigest()[:8]}. Use apenas o texto para responder.\n" + " ".join(sentences) + "\n" + question
                add(prompt, answer(facts, field))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    base, fingerprint = load_instructions(args.base)
    # Sem novos alvos públicos ruidosos; preserva documentos e validação autoral anteriores.
    rows = [r for r in base if r["task"].startswith("grounded_") or r["split"] == "validation"]
    rows += balanced_replay(base, 160)
    added = build()
    rows += added
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    load_instructions(path)
    manifest = dict(base_sha256=fingerprint, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    splits=dict(Counter(r["split"] for r in rows)), new_families=dict(Counter(r["task"] for r in added)),
                    note="Casos autorais controlados. Validação por formas e nomes diferentes, não teste cego nem prova de conversa geral. Nenhum teste consumido. Pesos herdam fontes anteriores.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
