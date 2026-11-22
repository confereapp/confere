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
from prepare_natural_answers import STATEMENTS, FEATURES
from prepare_natural_answers import QUESTIONS as NATURAL_QUESTIONS
from evaluate_context_smoke import CASES
from prepare_lexical_binding import FORMS, WORDS, expected
from expand_context_answers import names
from prepare_api_batches import local_rows, scene_for, PROFILES

DOC_FORMS = FORMS + [
    ("Há evidência explícita de {}.", "Há indicação explícita de ausência de {}."),
    ("A versão entregue contém {}.", "A versão entregue não contém {}."),
    ("A revisão confirmou {} no projeto.", "A revisão confirmou que o projeto está sem {}."),
    ("Foi demonstrado o funcionamento de {}.", "Foi demonstrado que falta {}."),
    ("O recurso {} está implementado.", "O recurso {} não está implementado."),
    ("Existe suporte a {}.", "Não existe suporte a {}."),
]
DOMAIN_PAIRS = [("importação", "exportação"), ("autenticação", "autorização"), ("cadastro", "recadastro"), ("criptografia", "descriptografia"), ("upload", "download"), ("inclusão", "exclusão"), ("ativação", "desativação"), ("validação", "invalidação"), ("conexão", "desconexão"), ("instalação", "desinstalação")]
PREFIXES = ["im", "ex", "re", "de", "in", "con", "trans", "auto", "micro", "macro", "sub", "inter", "super", "pré", "pós"]
QUESTIONS = {
    "nome": ["Que nome consta para o projeto?", "Qual o nome que este projeto recebeu?", "O projeto foi chamado de quê?"],
    "responsavel": ["Quem consta como responsável?", "Qual pessoa assume a responsabilidade?", "Quem está encarregado do projeto?"],
    "prazo": ["Quantos dias constam para a entrega?", "Qual quantidade de dias foi indicada?", "Em quantos dias deve ser entregue?"],
    "linguagem": ["Que linguagem consta no registro?", "Em qual linguagem foi programado?", "O texto apresenta qual linguagem?"],
}
for field in QUESTIONS:
    QUESTIONS[field]=list(dict.fromkeys(QUESTIONS[field]+NATURAL_QUESTIONS[field]))
QUESTIONS['prazo']+=['Quantos dias foram informados?','Quantos dias foram mencionados?','Qual duração consta no texto?']


def conversations(count=2400):
    rng = random.Random(161130); rows = []
    for i in range(count):
        a, b = rng.sample(NAMES, 2)
        facts = dict(nome=[rng.choice(["Serra", "Brisa", "Vale", "Trilha", "Sol", "Cedro"])],
                     responsavel=[a], prazo=[rng.randint(1, 180)], linguagem=[rng.choice(["Python", "Java", "Lua", "Ruby", "Go", "C", "Rust"])])
        retained=rng.sample(list(facts),rng.randint(1,len(facts)))
        facts={key:value for key,value in facts.items() if key in retained}
        changed = rng.choice(list(facts)); mode = i % 4
        if mode == 1: del facts[changed]
        elif mode == 2:
            alternatives=dict(nome=["Serra", "Brisa", "Vale", "Trilha", "Sol", "Cedro"],responsavel=[b],
                              prazo=list(range(1,181)),linguagem=["Python", "Java", "Lua", "Ruby", "Go", "C", "Rust"])
            facts[changed].append(rng.choice([v for v in alternatives[changed] if v!=facts[changed][0]]))
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
                             history=[], turn_index=0, response_tokens=256, prompt=header+text+rng.choice(["\n"," ","\nPergunta: "])+question, target=target,
                             scenario=dict(facts=copy.deepcopy(facts), field=field),
                             provenance=dict(kind="synthetic_authored", note="Perguntas contrastivas sobre os mesmos fatos; somente treino de desenvolvimento.")))
    for i in range(count // 2):
        first, second = rng.sample(NAMES, 2)
        group = f"query-memory-{i}"; history = []
        turns = [
            (rng.choice(["Pode me chamar pelo nome {}.", "Eu me chamo {}.", "Meu nome é {}."]).format(first), f"Prazer, {first}."),
            (rng.choice(["Na verdade, eu me chamo {}.", "Retificando, me chamo {}.", "Preciso corrigir: me chamo {}.", "Eu disse errado: meu nome é {}."]).format(second), f"Prazer, {second}."),
            (rng.choice(["Qual meu nome depois de corrigir?", "Agora como eu me chamo?", "Qual foi o nome corrigido?", "Que nome devo usar depois da correção?", "Qual nome devo usar para me identificar?"]), f"Seu nome é {second}."),
        ]
        for j, (prompt, target) in enumerate(turns):
            rows.append(dict(id=f"{group}-{j}", project_id=group, split="train", task="conversation_context_answer_memory",
                             prompt=prompt, target=target, history=copy.deepcopy(history), turn_index=j, response_tokens=256,
                             provenance=dict(kind="synthetic_authored", note="Nomes autorais de treino; correções explícitas com formas variadas. Desenvolvimento, sem teste cego.")))
            history.append([prompt, target])
    return rows



def comparison_pairs(count=8000):
    rng=random.Random(161135);rows=[]
    values=dict(nome=['Serra','Brisa','Vale','Trilha','Sol','Cedro'],responsavel=NAMES,
                prazo=list(range(1,181)),linguagem=['Python','Java','Lua','Ruby','Go','C','Rust'])
    for i in range(count):
        field=list(values)[i%4];first,second=rng.sample(values[field],2)
        others=[k for k in values if k!=field];rng.shuffle(others)
        background={k:[rng.choice(values[k])] for k in others[:rng.randrange(4)]}
        forms={k:rng.choice(STATEMENTS[k]+[FIELDS[k][0]]) for k in values}
        question=rng.choice(QUESTIONS[field]+FIELDS[field][1]);separator=rng.choice([' ','\n','\nPergunta: '])
        header=rng.choice(['','Use apenas o texto para responder.\n','Contexto: '])
        for j,items in enumerate([[first],[first,first],[first,second]]):
            facts=dict(background,**{field:items})
            sentences=[forms[k].format(v) for k,vs in facts.items() for v in vs];rng.shuffle(sentences)
            group=f'comparison-{i}-{j}';kind='conflict' if j==2 else 'fact'
            rows.append(dict(id=group,project_id=group,split='train',task='conversation_context_answer_'+kind,
                             prompt=header+' '.join(sentences)+separator+question,target=answer(facts,field),history=[],turn_index=0,response_tokens=256,
                             scenario=dict(facts=copy.deepcopy(facts),field=field,comparison_group=f'comparison-{i}'),
                             provenance=dict(kind='synthetic_authored',note='Mesmo valor isolado, repetido e contraposto a outro valor da mesma distribuição. Apenas treino.')))
    return rows


def scoped_conversations(count=4000):
    rng=random.Random(171101);rows=[]
    def add(group,prompt,target,scenario):
        if 'positive' in scenario:kind='support'
        elif not scenario['facts'].get('prazo'):kind='missing'
        elif len(set(scenario['facts']['prazo']))>1:kind='conflict'
        else:kind='fact'
        rows.append(dict(id=group,project_id=group,split='train',task='conversation_context_answer_'+kind,
                         prompt=prompt,target=target,history=[],turn_index=0,response_tokens=256,scenario=scenario,
                         provenance=dict(kind='synthetic_authored',note='Escopo da consulta com números distratores ou recursos compostos variados; somente treino de desenvolvimento.')))
    for i in range(count):
        def number():return rng.randint(1,20 if rng.random()<.5 else 180)
        a=number();b=number()
        while a==b:b=number()
        noise=[rng.choice(['A equipe tem {} integrantes.','O relatório contém {} páginas.','Foram planejadas {} reuniões.','O código tem {} módulos.']).format(number()) for _ in range(rng.randint(1,4))]
        extras=[rng.choice(STATEMENTS['linguagem']).format(rng.choice(['Ruby','Java','Python','Lua'])),
                rng.choice(STATEMENTS['nome']).format(rng.choice(['Serra','Vale','Trilha'])),
                rng.choice(STATEMENTS['responsavel']).format(rng.choice(NAMES))]
        rng.shuffle(extras);noise+=extras[:rng.randrange(4)]
        form=rng.choice(STATEMENTS['prazo']+[FIELDS['prazo'][0]]);question=rng.choice(QUESTIONS['prazo']+FIELDS['prazo'][1]);separator=rng.choice([' ','\n'])
        for j,values in enumerate([[],[a],[a,a],[a,b]]):
            sentences=noise+[form.format(v) for v in values];rng.shuffle(sentences);facts=dict(prazo=values)
            add(f'numeric-scope-{i}-{j}',' '.join(sentences)+separator+question,answer(facts,'prazo'),dict(facts=facts,field='prazo',noise=list(noise)))
        left,right=[v.casefold() for v in names(rng)];x,y=[v.casefold() for v in names(rng)]
        if i%3==0:
            left,right=rng.sample(['cadastro','consulta','relatórios','pagamentos','filtros','alertas','edição','histórico','avaliações','reservas','gestão','controle','monitoramento','exportação','importação','sincronização'],2)
            x,y=rng.sample(['clientes','produtos','compras','tarefas','mensagem','turmas','documentos','serviços','contas','agendamentos','matrículas','fornecedores','cartão','busca','acessos','comentários','salas','lote','nuvem','planilhas'],2)
        prepositions=[rng.choice(['de','por','para','com','em']) for _ in range(3)]
        features=[left+' '+prepositions[0]+' '+x,left+' '+prepositions[1]+' '+y,right+' '+prepositions[2]+' '+x];positive,negative,missing=rng.sample(features,3)
        verb=rng.choice(['inclui','contém','tem','possui','fornece'])
        questions=[rng.choice(['O projeto inclui {}?','Esse projeto tem {}?','Há {} no projeto?','O texto confirma a presença de {}?']).format(feature) for feature in features]
        for swap in range(2):
            yes,no=(positive,negative) if not swap else (negative,positive)
            statements=[f'O projeto {verb} {yes}.',f'O projeto não {verb} {no}.'];rng.shuffle(statements)
            text=' '.join(statements)
            if i%3==0:text=f'O projeto {verb} {yes}, mas não {verb} {no}.'
            if i%3==1:text=f'O projeto não {verb} {no}, mas {verb} {yes}.'
            if i%2:text+=f' A equipe discutiu {missing}, sem confirmar a implementação.'
            for j,feature in enumerate(features):
                target='Sim.' if feature==yes else 'Não.' if feature==no else 'Não tenho informação suficiente para responder.'
                add(f'open-scope-{i}-{swap}-{j}',text+separator+questions[j],target,dict(positive=yes,negative=no,queried=feature,missing=missing,text=text,prepositions=prepositions,paired_presence_group=f'open-scope-{i}-{swap}'))
    return rows


def presence_pairs(count=1600):
    rng = random.Random(161134); rows = []
    excluded = {c["message"] for c in CASES}
    questions = ["O registro confirma a presença de {}?", "A parte de {} foi incluída?", "O item {} foi incluído?", "Há {} nesse projeto?"]
    for i in range(count):
        a, b, missing = rng.sample(FEATURES, 3)
        clauses = [f"O projeto inclui {a}.", f"O projeto não inclui {b}."]
        rng.shuffle(clauses); text = " ".join(clauses)
        if i % 3 == 0:
            text = (f"O projeto inclui {a}, mas não inclui {b}." if rng.random() < .5 else
                    f"O projeto não inclui {b}, mas inclui {a}.")
        for j, (feature, target) in enumerate([(a, "Sim."), (b, "Não."), (missing, "Não tenho informação suficiente para responder.")]):
            group = f"query-presence-{i}-{j}"
            question = rng.choice(questions).format(feature)
            if rng.random() < .5:
                question = (f"Os {feature} foram incluídos?" if feature == "testes" else
                            f"A {feature} foi incluída?" if feature == "documentação" else f"O {feature} foi incluído?")
            prompt = text + " " + question
            if prompt in excluded:
                prompt = text + " " + questions[0].format(feature)
            assert prompt not in excluded
            rows.append(dict(id=group, project_id=group, split="train", task="conversation_context_answer_support",
                             prompt=prompt, target=target, history=[], turn_index=0, response_tokens=256,
                             scenario=dict(text=text, present=a, absent=b, missing=missing, queried=feature),
                             provenance=dict(kind="synthetic_authored", note="Mesmo texto com perguntas de presença opostas e atributo ausente; não copia diagnósticos completos.")))
    return rows

def documents(count=3600, *, topics=None, forms=None, sentence_units=False):
    rng = random.Random(161131); rows = []
    for i in range(count):
        a, b = [s.casefold() for s in names(rng)]
        if i % 3 == 0:
            a = rng.choice(WORDS); b = a + rng.choice([" adicional", " externo", " antigo"])
        elif i % 3 == 1:
            stem = a; left, right = rng.sample(PREFIXES, 2); a, b = left+stem, right+stem
        if i % 4 == 0: a, b = rng.choice(DOMAIN_PAIRS)
        if topics is not None: a,b=rng.sample(topics,2)
        pos, neg = rng.choice(DOC_FORMS if forms is None else forms)
        first, second = rng.sample(["atendido", "nao_atendido"], 2)
        states = [(a, first), (b, second)]; mode = rng.randrange(6)
        if mode == 1: states = [(a, second), (b, first)]
        if mode == 2: states.append((a, second))
        if mode == 3: states = [(b, second)]
        if mode == 4: states = []
        rng.shuffle(states)
        filler = " A equipe registrou reuniões e distribuiu as tarefas da semana." * rng.randrange(7)
        text = " ".join((pos if status == "atendido" else neg).format(topic) for topic, status in states)
        if not states: text = f"A equipe discutiu {a}; ainda não há comprovação de implementação."
        text = (text+filler if i % 2 else filler+" "+text).strip() + f" Registro técnico {rng.getrandbits(48):012x}."
        units = [dict(text=text, facts=states)]
        if sentence_units:
            units=[dict(text=(pos if status=='atendido' else neg).format(topic),facts=[(topic,status)]) for topic,status in states]
            units.append(dict(text=filler.strip()+f" Registro técnico {rng.getrandbits(48):012x}.",facts=[]))
        aliases = [dict(id=f"page:{j+1}", text=u['text']) for j,u in enumerate(units)]
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



def long_documents(tokenizer, count=120):
    rng = random.Random(161132); rows = []; skipped = 0
    for i in range(count):
        for profile in PROFILES:
            topics = None
            if i % 3:
                a, b = [s.casefold() for s in names(rng)]
                suffix = rng.choice(["ção", "mento", "dade", "ncia", "tura", "grafia"])
                left, right = rng.sample(PREFIXES, 2)
                topics = [a+suffix, b+suffix] if i % 3 == 1 else [left+a+suffix, right+a+suffix]
            if i % 4 == 0: topics = rng.choice(DOMAIN_PAIRS)
            scene = scene_for("train", 700000+i, profile, rng, forms=DOC_FORMS, topics=topics)
            local = local_rows(scene, tokenizer)
            if local is None:
                skipped += 1
                continue
            for row in local:
                row.update(id="query-long-"+row["id"], project_id="query-long-"+row["project_id"])
            rows.extend(local)
    return rows, skipped


def intent_variations(source, count=3000):
    rng = random.Random(161133); rows = []
    pool = [r for r in source if r["split"] == "train" and r["task"].startswith("conversation_")
            and not r["task"].startswith("conversation_context_answer_")]
    excluded = {r["prompt"].strip().casefold() for r in source if r["split"] != "train"}
    replacements = [("o que", "o q"), ("oq", "o q"), ("você", "vc"), ("você", "tu"),
                    ("vc", "c"), ("não", "nao"), ("estou", "tô"), ("ainda", "até agora"),
                    ("não sei", "estou em dúvida"), ("não sei", "não decidi"), ("como", "como é que")]
    for i in range(count):
        original = rng.choice(pool); prompt = original["prompt"]
        rng.shuffle(replacements)
        for a, b in replacements[:rng.randint(2, 7)]: prompt = prompt.replace(a, b)
        if prompt == original["prompt"] or prompt.strip().casefold() in excluded: continue
        row = copy.deepcopy(original); group = f"intent-variation-{i}"
        row.update(id=group, project_id=group, turn_index=0, prompt=prompt)
        rows.append(row)
    intents = {
        "hello": (["opa, podemos trocar ideia?", "opa, vamos bater papo?", "salve, vamos conversar?", "ei, bora trocar ideia?", "oi, podemos bater um papo?"], "Olá! Sou o Confere."),
        "function": (["o q tu faz?", "o q c faz aí?", "o q você consegue fazer?", "me diz o q tu faz aí", "qual é tua função aí?"], "Sou um experimento para apoiar a conferência de projetos."),
        "praise": (["vlw por explicar", "vlw pela resposta", "valeu pela explicação", "obrigado pela explicação", "valeu por explicar isso"], "Valeu! Se quiser, podemos continuar."),
        "negation": (["to sem projeto agr", "tô sem nenhum trabalho de projeto", "não tenho nenhum projeto agr", "ainda to sem projeto", "por enquanto tô sem um projeto"], "Sem problema. Podemos conversar sem um projeto."),
    }
    for i in range(300):
        for task, (prompts, target) in intents.items():
            prompt = rng.choice(prompts)
            assert prompt.casefold() not in excluded
            group = f"intent-colloquial-{task}-{i}"
            rows.append(dict(id=group, project_id=group, split="train", task="conversation_slang_"+task,
                             prompt=prompt, target=target, history=copy.deepcopy(rng.choice(pool)["history"][-3:]), turn_index=0,
                             response_tokens=256, provenance=dict(kind="synthetic_authored", note="Intenção atual em históricos variados; formas de desenvolvimento sem copiar pedidos da validação.")))
    for i in range(600):
        group = f"intent-state-topic-{i}"; available = bool(i % 2)
        seed = copy.deepcopy(rng.choice(pool)["history"][-rng.randint(2, 5):])
        update = rng.choice(["retificando, agora já tenho um projeto", "corrigindo a situação: já tenho projeto", "atualização: tenho um projeto agora"]) if available else rng.choice(["retificando, continuo sem ter um projeto", "corrigindo a situação: ainda não tenho projeto", "atualização: por enquanto estou sem projeto"])
        acknowledgement = "Entendi: agora você tem um projeto." if available else "Entendi: você ainda não tem um projeto."
        state = "Você disse que tem um projeto." if available else "Você disse que ainda não tem um projeto."
        topic = rng.choice(["música", "cinema", "jogos", "livros", "estudos", "viagens"])
        turns = [("state", update, acknowledgement),
                 ("state", rng.choice(["qual é a situação atual do projeto?", "me lembra como está a situação do meu projeto", "o que eu disse sobre ter um projeto?"]), state),
                 ("topic", rng.choice(["quero que o assunto seja {}", "o tema que escolhi para conversar é {}", "vamos bater papo a respeito de {}"]).format(topic), f"Podemos conversar sobre {topic}."),
                 ("topic", rng.choice(["do que estamos falando agora?", "do que estamos conversando neste momento?", "sobre que tema estamos conversando neste momento?", "qual é o assunto da nossa conversa agora?"]), f"Estamos conversando sobre {topic}.")]
        for j, (task, prompt, target) in enumerate(turns):
            assert prompt.casefold() not in excluded
            rows.append(dict(id=f"{group}-{j}", project_id=group, split="train", task="conversation_flow_"+task,
                             prompt=prompt, target=target, history=copy.deepcopy(seed), turn_index=j, response_tokens=256,
                             provenance=dict(kind="synthetic_authored", note="Mudanças de estado e tema em posições variadas; somente treino de desenvolvimento.")))
            seed.append([prompt, target])
    # A mesma incerteza pede uma proposta no início e foco após as sugestões.
    proposal = "Sem problema. Podemos falar de objetivos, requisitos ou evidências."
    focus = "Vamos começar pelo tema do seu projeto. Qual é o tema?"
    uncertain = ["não tenho certeza do que pedir agora", "estou em dúvida sobre o que posso pedir", "não decidi como você pode ajudar", "não sei escolher o que pedir a você", "ainda não decidi o que vou pedir"]
    again = ["continuo indeciso entre as sugestões", "ainda tenho dúvida entre essas alternativas", "não consigo escolher dentre essas opções", "mesmo depois das sugestões continuo em dúvida", "nenhuma dessas opções me ajudou a decidir", "não sei qual das sugestões faz mais sentido", "ainda não sei qual alternativa faz sentido pra mim", "não consigo saber qual opção faz sentido"]
    for i in range(300):
        first = rng.choice(uncertain); second = rng.choice(again)
        seed = copy.deepcopy(rng.choice(pool)["history"][-2:]); group = f"intent-guidance-{i}"
        for j, (prompt, target) in enumerate([(first, proposal), (second, focus)]):
            rows.append(dict(id=f"{group}-{j}", project_id=group, split="train", task="conversation_guidance",
                             prompt=prompt, target=target, history=copy.deepcopy(seed), turn_index=j, response_tokens=256,
                             provenance=dict(kind="synthetic_authored", note="Variação autoral de intenção com histórico; somente treino.")))
            seed.append([prompt, target])
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
    skipped = 0
    if args.task == "conversation": added = conversations() + presence_pairs() + intent_variations(source)
    else:
        long_rows, skipped = long_documents(tokenizer)
        added = documents() + long_rows
    duplicates = 0
    if args.task == "documents":
        known = {r["prompt"].strip().casefold(): r["target"] for r in source}
        unique = []
        for row in added:
            key = row["prompt"].strip().casefold()
            if key in known:
                if json.loads(known[key]) != json.loads(row["target"]): raise ValueError("Fragmento com alvos conflitantes.")
                duplicates += 1
                continue
            known[key] = row["target"]; unique.append(row)
        added = unique
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
                    splits=dict(Counter(r["split"] for r in rows)), development_rows=len(development), skipped_partial_evidence_scenes=skipped, duplicate_train_fragments_omitted=duplicates,
                    development_cases_sha256=hashlib.sha256(args.development_cases.read_bytes()).hexdigest() if args.development_cases else None,
                    note="Validação original preservada. Coorte antes reservada agora declarada desenvolvimento após falha; nunca entrou no treino. Novo teste pós-congelamento obrigatório.")
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
