"""Variações autorais de fatos, tarefas e presença; somente treino é ampliado."""
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
from prepare_context_answers import answer, ABSENT, CONFLICT
from evaluate_context_smoke import CASES

HELD_NAMES = {"Clara", "Davi", "Renata", "Victor", "Lia", "Rui"}
STATEMENTS = {
    "nome": ["O projeto recebeu o nome {}.", "Este projeto é chamado {}.", "O nome registrado para o projeto é {}."],
    "responsavel": ["A responsável pelo projeto é {}.", "A responsabilidade ficou com {}.", "{} responde pelo projeto.", "O registro aponta {} como responsável."],
    "prazo": ["A entrega está prevista para daqui a {} dias.", "Faltam {} dias para entregar.", "O prazo de entrega é de {} dias.", "O texto informa {} dias para a entrega."],
    "linguagem": ["O projeto usa {}.", "A implementação utiliza {}.", "A linguagem escolhida foi {}.", "A equipe programa em {}."],
}
QUESTIONS = {
    "nome": ["Qual é o nome desse projeto?", "Como se chama esse projeto?", "Diga o nome registrado para o projeto."],
    "responsavel": ["Quem responde pelo projeto?", "Qual é o nome da pessoa responsável?", "Quem ficou com a responsabilidade?"],
    "prazo": ["Qual é o prazo informado?", "Em quantos dias será a entrega?", "Quanto tempo falta para entregar?", "Qual prazo o texto apresenta?"],
    "linguagem": ["Qual linguagem ele usa?", "Em qual linguagem a equipe programa?", "Qual linguagem foi escolhida?"],
}
FEATURES = ["testes", "documentação", "backup", "cadastro", "login", "cronograma"]
ROLES = [("testes", "{} faz os testes.", "Quem faz os testes?"),
         ("manual", "{} escreve o manual.", "Quem escreve o manual?"),
         ("interface", "{} prepara a interface.", "Quem prepara a interface?"),
         ("banco", "{} organiza o banco de dados.", "Quem organiza o banco de dados?")]


def name(rng):
    while True:
        value = (rng.choice(["Iara", "Noa", "Teo", "Caê", "Luiz", "Inês", "Dora", "Hugo", "Íris", "Bia", "Ari", "Ítalo"])
                 if rng.random() < .5 else "".join(rng.choices("abcdeghijlmnoprstuvzáéíóúãç", k=rng.randint(2, 9))).capitalize())
        if value not in HELD_NAMES: return value


def build(source, contexts=4000, roles=1000, presence=1200):
    rows = list(source); rng = random.Random(161116)
    excluded = {case["message"] for case in CASES}
    def add(task, prompt, target, scenario):
        if prompt in excluded: raise ValueError("Diagnóstico completo não pode entrar no treino.")
        group = f"natural-answer-{len(rows)}"
        rows.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_" + task,
                         history=[], turn_index=0, response_tokens=256, prompt=prompt, target=target, scenario=scenario,
                         provenance=dict(kind="synthetic_authored", note="Fatos e respostas derivados por construção; variação somente de treino, sem revisão independente.")))
    for i in range(contexts):
        first = name(rng)
        facts = dict(nome=[name(rng)], responsavel=[first], prazo=[rng.randint(1, 120)],
                     linguagem=[rng.choice(["Python", "Java", "Lua", "Ruby", "Go", "C", "Rust"])])
        field = rng.choice(list(facts)); state = i % 3
        if state == 1: del facts[field]
        elif state == 2:
            alternate = (str(facts[field][0]) + "z" if field in {"nome", "responsavel"} else
                         200 if field == "prazo" else "Pascal")
            facts[field].append(alternate)
        sentences = [rng.choice(STATEMENTS[key]).format(value) for key, values in facts.items() for value in values]
        rng.shuffle(sentences)
        text = " ".join(sentences); question = rng.choice(QUESTIONS[field])
        prompt = rng.choice([text + " " + question, "Leia: " + text + "\n" + question, question + "\n" + text])
        add(["fact", "missing", "conflict"][state], prompt, answer(facts, field), dict(facts=facts, field=field))
    for _ in range(roles):
        first = name(rng); second = name(rng)
        while second == first: second = name(rng)
        a, b = rng.sample(ROLES, 2)
        assignments = {a[0]: first, b[0]: second}; requested = rng.choice([a, b])
        sentences = [a[1].format(first), b[1].format(second)]; rng.shuffle(sentences)
        prompt = rng.choice(["Equipe: ", "Tarefas: ", ""]) + " ".join(sentences) + " " + requested[2]
        add("fact", prompt, assignments[requested[0]] + ".", dict(assignments=assignments, requested=requested[0]))
    for i in range(presence):
        feature, other, missing = rng.sample(FEATURES, 3); state = i % 4
        if state == 0: text = f"O projeto inclui {feature}, mas não inclui {other}."; queried = feature; target = "Sim."
        elif state == 1: text = f"O projeto não inclui {feature}, mas inclui {other}."; queried = feature; target = "Não."
        elif state == 2: text = f"O projeto inclui {other}."; queried = missing; target = ABSENT
        else: text = f"O relatório diz que há {feature} e depois diz que não há {feature}."; queried = feature; target = CONFLICT
        question = rng.choice(["Há {} no projeto?", "O projeto inclui {}?", "O texto confirma que há {}?", "Dá para confirmar a presença de {}?"]).format(queried)
        add("conflict" if state == 3 else "support", text + " " + question, target, dict(state=state, queried=queried))
    assert [r for r in rows if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source); rows = build(source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True); path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    splits=dict(Counter(r["split"] for r in rows)), added=len(rows)-len(source), held_names=sorted(HELD_NAMES),
                    note="Validação original intacta. Nenhum diagnóstico completo copiado; casos de desenvolvimento, não teste cego.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
