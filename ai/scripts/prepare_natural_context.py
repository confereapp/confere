"""Contrastes de respondibilidade e realizações naturais dos mesmos fatos autorais."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import torch
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_context_answers import answer
from expand_context_answers import names
from prepare_api_batches import scene_for, local_rows

STATEMENTS = {
    "nome": ["O projeto se chama {}.", "O nome do projeto é {}.", "O projeto recebeu o nome {}."],
    "responsavel": ["A pessoa responsável é {}.", "Quem responde pelo projeto é {}.", "{} é a pessoa responsável pelo projeto."],
    "prazo": ["O prazo informado é de {} dias.", "A entrega está prevista para daqui a {} dias.", "Há {} dias para a entrega."],
    "linguagem": ["A linguagem usada é {}.", "O projeto usa {}.", "O desenvolvimento será feito em {}."],
}
QUESTIONS = {
    "nome": ["Como o projeto foi chamado?", "Informe o nome mencionado para o projeto.", "Que nome o projeto recebeu?"],
    "responsavel": ["Quem ficou responsável?", "Diga quem responde por esse projeto.", "Quem assumiu a responsabilidade?"],
    "prazo": ["Quantos dias o texto menciona?", "Quantos dias há para concluir?", "Diga a quantidade de dias mencionada."],
    "linguagem": ["Que linguagem o projeto utiliza?", "Qual tecnologia de programação foi mencionada?", "Diga a linguagem utilizada."],
}


def render(facts, question, rng):
    sentences = [rng.choice(STATEMENTS[key]).format(value) for key, values in facts.items() for value in values]
    rng.shuffle(sentences)
    text = " ".join(sentences)
    return rng.choice([text+"\n"+question, "Contexto: "+text+"\nPergunta: "+question,
                       question+"\n"+text, "Leia o texto e responda: "+text+"\n"+question])


def build(source, paired=2000, role_count=500, presence=500):
    result = []
    for original in source:
        row = copy.deepcopy(original)
        if row["split"] == "train" and row["task"].startswith("conversation_context_answer_") and not row["task"].endswith(("memory", "support")):
            rng = random.Random(int(hashlib.sha256(row["id"].encode()).hexdigest(), 16))
            question = row["prompt"].splitlines()[-1]
            row["prompt"] = render(row["scenario"]["facts"], question, rng)
            row["id"] = "natural-"+row["id"]; row["project_id"] = "natural-"+row["project_id"]
        result.append(row)
    rng = random.Random(151128)
    for i in range(paired):
        first, _ = names(rng)
        facts = dict(nome=[rng.choice(["Atlas", "Ponte", "Farol", "Jardim"])], responsavel=[first],
                     prazo=[rng.randint(1, 90)], linguagem=[rng.choice(["Python", "Java", "C", "Ruby"])])
        missing = list(facts)[i % 4]
        del facts[missing]
        present = rng.choice(list(facts))
        # Mesmos fatos, perguntas de campos diferentes; cada alvo é derivado do campo consultado.
        for field in [present, missing]:
            group = f"answerability-{i}-{field}"
            question = rng.choice(QUESTIONS[field])
            result.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_"+("missing" if field==missing else "fact"),
                               history=[], turn_index=0, response_tokens=256, prompt=render(facts, question, rng), target=answer(facts, field),
                               scenario=dict(facts=copy.deepcopy(facts), field=field),
                               provenance=dict(kind="synthetic_authored", note="Contraste de campo presente/ausente, alvo derivado dos fatos, sem revisão humana independente.")))
    for i in range(role_count):
        first, second = names(rng)
        task, other = rng.sample(["testes", "manual", "interface", "banco de dados"], 2)
        group = f"role-answer-{i}"
        result.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_fact",
                           history=[], turn_index=0, response_tokens=256,
                           prompt=f"{first} cuida de {task}; {second} cuida de {other}. Quem cuida de {task}?",
                           target=first+".", scenario=dict(roles={task:first, other:second}, requested=task),
                           provenance=dict(kind="synthetic_authored", note="Atribuição autoral de tarefas; resposta literal verificável, sem revisão humana independente.")))
    for i in range(presence):
        positive, negative, unknown = rng.sample(["testes", "documentação", "backup", "cronograma", "cadastro", "login"], 3)
        text = f"O projeto inclui {positive}, mas não inclui {negative}."
        for feature, target in [(positive, "Sim."), (negative, "Não."), (unknown, "Não tenho informação suficiente para responder.")]:
            group = f"presence-answer-{i}-{feature}"
            question = rng.choice(["Há {} no projeto?", "O texto confirma a presença de {}?", "O projeto tem {}?"]).format(feature)
            result.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_support", history=[], turn_index=0, response_tokens=256,
                               prompt=text+"\n"+question, target=target,
                               scenario=dict(positive=positive, negative=negative, queried=feature),
                               provenance=dict(kind="synthetic_authored", note="Presença, ausência explícita e informação não fornecida para requisitos diferentes no mesmo texto, sem revisão humana independente.")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source)
    rows = build(source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    rng = random.Random(151129)
    added_documents = 0
    for i in range(1000):
        scene = scene_for("train", 100000+i, "scope_short" if i%2 else "short_conflict", rng)
        for row in local_rows(scene, tokenizer, checkpoint["config"]["block_size"]) or []:
            row["id"] = "natural-doc-"+row["id"]; row["project_id"] = "natural-doc-"+row["project_id"]
            rows.append(row); added_documents += 1
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    # A validação permanece literalmente idêntica à versão B.
    assert [r for r in rows if r["split"]=="validation"] == [r for r in source if r["split"]=="validation"]
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), splits=dict(Counter(r["split"] for r in rows)),
                    paired_scenes=2000, role_cases=500, presence_scenes=500, additional_document_rows=added_documents,
                    note="Dados autorais, sem novas fontes públicas. Validação B preservada. Diagnósticos já inspecionados motivaram a variedade de formatos; desenvolvimento, não teste cego.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
