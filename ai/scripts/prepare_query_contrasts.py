"""Contrastes autorais: a pergunta muda, o texto permanece idêntico."""
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
from prepare_context_answers import answer, FIELDS, NAMES
from prepare_natural_answers import STATEMENTS
from prepare_lexical_binding import FORMS, WORDS, expected
from expand_context_answers import names
from prepare_api_batches import local_rows

DOC_FORMS = FORMS + [
    ("Há evidência explícita de {}.", "Há indicação explícita de ausência de {}."),
    ("A versão entregue contém {}.", "A versão entregue não contém {}."),
    ("A revisão confirmou {} no projeto.", "A revisão confirmou que o projeto está sem {}."),
    ("Foi demonstrado o funcionamento de {}.", "Foi demonstrado que falta {}."),
    ("O recurso {} está implementado.", "O recurso {} não está implementado."),
    ("Existe suporte a {}.", "Não existe suporte a {}."),
]
QUESTIONS = {
    "nome": ["Que nome consta para o projeto?", "Qual o nome que este projeto recebeu?", "O projeto foi chamado de quê?"],
    "responsavel": ["Quem consta como responsável?", "Qual pessoa assume a responsabilidade?", "Quem está encarregado do projeto?"],
    "prazo": ["Quantos dias constam para a entrega?", "Qual quantidade de dias foi indicada?", "Em quantos dias deve ser entregue?"],
    "linguagem": ["Que linguagem consta no registro?", "Em qual linguagem foi programado?", "O texto apresenta qual linguagem?"],
}


def conversations(count=2400):
    rng = random.Random(161130); rows = []
    for i in range(count):
        a, b = rng.sample(NAMES, 2)
        facts = dict(nome=[rng.choice(["Serra", "Brisa", "Vale", "Trilha", "Sol", "Cedro"])],
                     responsavel=[a], prazo=[rng.randint(1, 180)], linguagem=[rng.choice(["Python", "Java", "Lua", "Ruby", "Go", "C", "Rust"])])
        changed = rng.choice(list(facts)); mode = i % 4
        if mode == 1: del facts[changed]
        elif mode == 2: facts[changed].append(dict(nome="Mar", responsavel=b, prazo=200, linguagem="Pascal")[changed])
        elif mode == 3: facts[changed].append(facts[changed][0])
        sentences = [(FIELDS[k][0] if i % 2 else rng.choice(STATEMENTS[k])).format(v) for k, vs in facts.items() for v in vs]
        rng.shuffle(sentences); text = " ".join(sentences)
        header = rng.choice(["", "Use apenas o texto para responder.\n", f"Registro {rng.getrandbits(32):08x}. Use apenas o texto para responder.\n"])
        for field in FIELDS:
            question = rng.choice(QUESTIONS[field] + FIELDS[field][1])
            target = answer(facts, field)
            family = "missing" if not facts.get(field) else "conflict" if len(set(facts[field])) > 1 else "fact"
            group = f"query-chat-{i}-{field}"
            rows.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_"+family,
                             history=[], turn_index=0, response_tokens=256, prompt=header+text+"\n"+question, target=target,
                             scenario=dict(facts=copy.deepcopy(facts), field=field),
                             provenance=dict(kind="synthetic_authored", note="Perguntas contrastivas sobre os mesmos fatos; somente treino de desenvolvimento.")))
    return rows


def documents(count=3600):
    rng = random.Random(161131); rows = []
    for i in range(count):
        a, b = [s.casefold() for s in names(rng)]
        if i % 3 == 0:
            a = rng.choice(WORDS); b = a + rng.choice([" adicional", " externo", " antigo"])
        elif i % 3 == 1:
            stem = a; a, b = "pré"+stem, "pós"+stem
        pos, neg = rng.choice(DOC_FORMS)
        states = [(a, "atendido"), (b, "nao_atendido")]
        if i % 6 == 1: states = [(a, "nao_atendido"), (b, "atendido")]
        if i % 6 == 2: states.append((a, "nao_atendido"))
        if i % 6 == 3: states = [(b, "nao_atendido")]
        if i % 6 == 4: states = []
        rng.shuffle(states)
        filler = " A equipe registrou reuniões e distribuiu as tarefas da semana." * rng.randrange(7)
        text = " ".join((pos if status == "atendido" else neg).format(topic) for topic, status in states)
        if not states: text = f"A equipe discutiu {a}; ainda não há comprovação de implementação."
        text = (text+filler if i % 2 else filler+" "+text).strip() + f" Registro técnico {rng.getrandbits(48):012x}."
        units = [dict(text=text, facts=states)]
        aliases = [dict(id="page:1", text=text)]
        for j, feature in enumerate([a, b]):
            target = expected(units, feature); requirement = f"O projeto deve incluir {feature}."
            group = f"query-doc-{i}-{j}"
            row = dict(id=group, project_id=f"query-doc-{i}", split="train", task="grounded_"+target["status"],
                       grounding_schema=2, evaluation_profile="query_contrast", requirement=requirement, units=copy.deepcopy(aliases),
                       document=f"query-{i}.txt", prompt=prompt_for(requirement, aliases, 2), target=json.dumps(target, ensure_ascii=False),
                       scenario=dict(units=copy.deepcopy(units), feature=feature),
                       provenance=dict(kind="synthetic_authored", note="Estados derivados por construção; nomes próximos e molduras variadas. Não é teste cego."))
            assert judge(row["target"], row)["exact_match"]
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path); p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--task", choices=["conversation", "documents"], required=True)
    p.add_argument("--development-cases", type=Path)
    args = p.parse_args()
    if args.output.exists(): p.error("Saída deve ser nova.")
    if args.development_cases and args.task != "documents": p.error("Casos completos são documentais.")
    source, digest = load_instructions(args.source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True); tokenizer = ByteBPE(checkpoint["merges"])
    added = conversations() if args.task == "conversation" else documents()
    rows = source + added
    assert [r for r in rows if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    development = []
    if args.development_cases:
        for scene in json.loads(args.development_cases.read_text()):
            local = local_rows(scene, tokenizer, checkpoint["config"]["block_size"])
            if local is None: raise ValueError("Evidência parcial não pode receber alvo inventado.")
            for row in local:
                row.update(id="development-"+row["id"], project_id="development-"+row["project_id"], split="validation",
                           evaluation_profile="development_"+row["evaluation_profile"])
            development.extend(local)
        rows += development
    validation_prompts = {r["prompt"] for r in rows if r["split"] != "train"}
    if any(r["prompt"] in validation_prompts for r in added): raise ValueError("Colisão com validação.")
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True); path = args.output/"examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), added_train=len(added),
                    splits=dict(Counter(r["split"] for r in rows)), development_rows=len(development),
                    development_cases_sha256=hashlib.sha256(args.development_cases.read_bytes()).hexdigest() if args.development_cases else None,
                    note="Validação original preservada. Coorte antes reservada agora declarada desenvolvimento após falha; nunca entrou no treino. Novo teste pós-congelamento obrigatório.")
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
