"""Segundo currículo: perguntas variadas, nomes combinatórios e contraste usuário/assistente."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import random
import torch
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_context_answers import FIELDS, answer

ALIASES = {
    "nome": ["o nome do projeto", "a identificação do projeto", "o nome escolhido para o projeto"],
    "responsavel": ["a pessoa responsável", "o nome de quem responde pelo projeto", "a pessoa que tem a responsabilidade"],
    "prazo": ["o prazo em dias", "o número de dias do prazo", "a duração do prazo"],
    "linguagem": ["a linguagem de programação", "a linguagem usada", "a linguagem escolhida"],
}
QUESTION_FRAMES = ["Informe {}.", "Qual é {}?", "Diga apenas {}.", "Identifique {} no texto.",
                   "Considerando o texto, indique {}.", "Quero saber {}."]
USER_QUESTIONS = ["Qual é meu nome agora?", "Como eu me chamo?", "Qual é o nome do usuário?", "Diga o meu nome.",
                  "Com essa atualização, qual é meu nome?", "Você lembra o nome que informei?"]
ASSISTANT_QUESTIONS = ["E qual é seu nome?", "Como você se chama?", "Qual o nome do assistente?"]
SYLLABLES = ["ma", "ri", "na", "lu", "ca", "vi", "ta", "ra", "so", "pe", "li", "no", "da", "be", "fa", "te"]
HELD_NAMES = {"Clara", "Davi", "Renata", "Victor"}


def names(rng):
    result = []
    while len(result) < 2:
        name = "".join(rng.choices(SYLLABLES, k=rng.choice([2, 3]))).capitalize()
        if name not in HELD_NAMES and name not in result: result.append(name)
    return result


def expand(source):
    rows = []
    groups = {}
    corrections = []
    for row in source:
        if row["split"] != "train" or not row["task"].startswith("conversation_context_answer_"):
            row = copy.deepcopy(row)
            if row["split"] == "validation" and row["prompt"].endswith("Qual nome aparece no texto?"):
                row["prompt"] = row["prompt"].removesuffix("Qual nome aparece no texto?") + "Qual é o nome atribuído ao projeto no texto?"
                corrections.append(row["id"])
            rows.append(row)
        else:
            groups.setdefault(row["project_id"], []).append(row)
    for group, original in groups.items():
        rng = random.Random(int(hashlib.sha256(group.encode()).hexdigest(), 16))
        first, second = names(rng)
        prefix = "expanded-" + group
        if original[0]["task"].endswith("memory"):
            recall = [
                (rng.choice(USER_QUESTIONS), f"Seu nome é {second}.", "user"),
                (rng.choice(ASSISTANT_QUESTIONS), "Sou o Confere.", "assistant"),
            ]
            rng.shuffle(recall)
            turns = [
                (rng.choice(["Meu nome é {}.", "Eu me chamo {}.", "Me chamo {}."]).format(first), f"Prazer, {first}."),
                (rng.choice(["Corrigindo: pode me chamar de {}.", "Atualização: meu nome é {}.", "Use {} como meu nome daqui em diante."]).format(second), f"Prazer, {second}."),
            ] + [(prompt, target) for prompt, target, _ in recall]
            history = []
            for i, (prompt, target) in enumerate(turns):
                row = copy.deepcopy(original[0])
                row.update(id=f"{prefix}-{i}", project_id=prefix, turn_index=i, prompt=prompt,
                           target=target, history=copy.deepcopy(history), scenario=dict(names=[first, second], question_role=recall[i-2][2] if i >= 2 else None))
                rows.append(row)
                history.append([prompt, target])
            continue
        row = copy.deepcopy(original[0])
        facts = row["scenario"]["facts"]
        if "responsavel" in facts:
            facts["responsavel"] = [first] if len(facts["responsavel"]) == 1 else [first, second]
        field = row["scenario"]["field"]
        if row["task"].endswith(("fact", "missing")):
            # Fatos de outro assunto não mudam a resposta à pergunta atual.
            unrelated = [key for key in facts if key != field]
            if len(unrelated) > 1 and rng.random() < .5:
                del facts[rng.choice(unrelated)]
            unrelated = [key for key in facts if key != field]
            if unrelated and rng.random() < .5:
                key = rng.choice(unrelated)
                alternate = dict(nome="Caminho", responsavel=second, prazo=100, linguagem="Lua")[key]
                facts[key].append(alternate)
        if field == "testes":
            sentences = ["A equipe está registrada.", "O projeto inclui documentação."]
            state = facts.get("testes", [])
            if state: sentences.append("O projeto inclui testes." if state[0] == "presente" else "O projeto não inclui testes.")
            question = rng.choice(["O projeto tem testes?", "Os testes estão incluídos?", "Existem testes nesse projeto?", "É possível afirmar que há testes?", "O texto diz que existem testes?"])
        else:
            sentences = [FIELDS[key][0].format(value) for key, values in facts.items() for value in values]
            question = rng.choice(QUESTION_FRAMES).format(rng.choice(ALIASES[field]))
        rng.shuffle(sentences)
        row.update(id=prefix+"-0", project_id=prefix,
                   prompt=f"Registro {hashlib.sha256(prefix.encode()).hexdigest()[:8]}. Use apenas o texto para responder.\n" + " ".join(sentences) + "\n" + question,
                   target=answer(facts, field))
        rows.append(row)
    # Contrastes adicionais de desenvolvimento: distratores não alteram o fato consultado.
    validation_facts = [r for r in rows if r["split"] == "validation" and r["task"].endswith("_fact")]
    for original in validation_facts:
        row = copy.deepcopy(original)
        row["id"] = "distractor-" + row["id"]
        row["project_id"] = "distractor-" + row["project_id"]
        facts = row["scenario"]["facts"]
        field = row["scenario"]["field"]
        key = next(key for key in facts if key != field)
        alternative = dict(nome="Caminho", responsavel="Outra pessoa", prazo=100, linguagem="Lua")[key]
        facts[key].append(alternative)
        row["scenario"]["source_id"] = original["id"]
        sentences = [FIELDS[key][0].format(value) for key, values in facts.items() for value in values]
        row["prompt"] = "Use apenas o texto para responder.\n" + " ".join(sentences) + "\n" + original["prompt"].splitlines()[-1]
        assert row["target"] == answer(facts, field)
        rows.append(row)
    return rows, corrections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source)
    rows, corrections = expand(source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    splits=dict(Counter(r["split"] for r in rows)), corrected_validation_ids=corrections, added_distractor_validation=12,
                    note="Validação versionada: pergunta de nome agora explicita projeto em contexto que também contém nome de pessoa. Comparar antes/depois no mesmo conjunto corrigido; preservar o relatório original. Formas e nomes de validação não entram no treino. Casos de desenvolvimento, não teste cego.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
