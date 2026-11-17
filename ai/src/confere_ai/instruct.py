"""Piloto SFT de tarefas curtas, inicializado exclusivamente do nosso GPT local."""
import argparse
import hashlib
import json
import math
import platform
import resource
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import GPT, GPTConfig
from .tokenizer import ByteBPE
from .training_data import encode_sample
from .train import batch, choose_device, evaluate, save_checkpoint
from .report import create_report, record_failure, save_samples
from .grounding import judge, prompt_for


def load_instructions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    projects, prompts, ids, contexts, conversations = {}, set(), set(), {}, {}
    for row in rows:
        human=row["provenance"]["kind"]=="human_openassistant"
        public=row["provenance"]["kind"]=="public_model_generated"
        nli=row['provenance']['kind']=='public_reviewed_nli'
        qa=row['provenance']['kind']=='public_extractive_qa'
        if (row['provenance']['kind']=='synthetic_authored'
                and row['task'].startswith('grounded_')
                and isinstance(row.get('scenario'),dict)
                and row['scenario'].get('criterion') in {'presence','absence','mention'}
                and row['provenance'].get('annotation_scope')!='component_inclusion_v2'):
            raise ValueError('Escopo de critério antigo ou não validado; regenere os contrastes de inclusão v2.')
        if qa:
            from .qa_source import validate_provenance
            validate_provenance(row)
        if nli:
            from .nli_source import validate_provenance
            validate_provenance(row)
        if public and (row["provenance"].get("source") not in {"https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v3", "https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft"} or row["provenance"].get("license")!="Apache-2.0" or not row["provenance"].get("revision")):
            raise ValueError("Proveniência pública não rastreável.")
        if human and (row["provenance"].get("source") not in {"https://huggingface.co/datasets/OpenAssistant/oasst1","https://huggingface.co/datasets/OpenAssistant/oasst2"} or row["provenance"].get("license")!="Apache-2.0" or not row["provenance"].get("revision")):
            raise ValueError("Proveniência humana não rastreável.")
        if row["split"] not in {"train", "validation", "test"} or row["provenance"]["kind"] not in {"synthetic_authored","human_openassistant","public_model_generated","public_reviewed_nli","public_extractive_qa"}:
            raise ValueError("Split ou proveniência inválida.")
        conversational = "history" in row
        if conversational:
            if not isinstance(row["history"],list) or any(not isinstance(p,list) or len(p)!=2 or any(not isinstance(t,str) for t in p) for p in row["history"]):
                raise ValueError("Histórico inválido.")
            key=json.dumps([row["history"],row["prompt"].strip().casefold()],ensure_ascii=False)
            value=(row["split"],row["target"])
            if contexts.setdefault(key,value)!=value: raise ValueError("Contexto duplicado entre splits ou alvos conflitantes.")
            previous=conversations.get(row["project_id"])
            if not (human or public) and ((previous is None and row["turn_index"]!=0) or (previous is not None and (row["turn_index"]!=previous["turn_index"]+1 or row["history"]!=previous["history"]+[[previous["prompt"],previous["target"]]]))):
                raise ValueError("Sequência de conversa inconsistente.")
            if not (human or public): conversations[row["project_id"]]=row
        if not row["prompt"].strip() or not row["target"].strip() or (not conversational and row["prompt"].strip().casefold() in prompts) or row["id"] in ids:
            raise ValueError("Texto vazio ou exemplos duplicados.")
        prompts.add(row["prompt"].strip().casefold()); ids.add(row["id"])
        if projects.setdefault(row["project_id"], row["split"]) != row["split"]:
            raise ValueError("Projeto atravessando splits.")
        if row["task"].startswith("grounded_"):
            if (row["prompt"] != prompt_for(row["requirement"], row["units"], row.get("grounding_schema", 1))
                or len({u["id"] for u in row["units"]}) != len(row["units"])
                or row["task"] != "grounded_" + json.loads(row["target"])["status"]
                or not judge(row["target"], row)["exact_match"]):
                raise ValueError("Alvo documental ou referência inválida.")
    if not all(any(row["split"] == split for row in rows) for split in ("train", "validation")):
        raise ValueError("Treino e validação são obrigatórios.")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def score(model, tokenizer, rows, device, context_limit=None, response_tokens=64):
    from .chat import prepare_prompt
    outputs, histories = [], {}
    for row in rows:
        trace={}
        if "history" in row:
            if row["turn_index"]==0: histories[row["project_id"]]=[tuple(p) for p in row["history"]]
            history=histories[row["project_id"]]
            ids,dropped=prepare_prompt(history,row["prompt"],tokenizer,context_limit or model.config.block_size,response_tokens,"conversation")
            trace={"history_used":[list(p) for p in history[dropped:]],"dropped_turns":dropped,"project_id":row["project_id"],"turn_index":row["turn_index"]}
        else:
            ids = [tokenizer.BOS] + tokenizer.encode(row["prompt"]) + [tokenizer.SEP]
        generated = model.generate(torch.tensor([ids], device=device), min(response_tokens, model.config.block_size-len(ids)), tokenizer.EOS)
        answer = tokenizer.decode(generated[0, len(ids):].tolist())
        if "history" in row: history.append((row["prompt"],answer))
        judged = judge(answer, row) if row["task"].startswith("grounded_") else {"exact_match": answer.strip() == row["target"].strip()}
        if "evaluation_profile" in row:
            trace["evaluation_profile"] = row["evaluation_profile"]
        outputs.append({"id": row["id"], "task": row["task"], "prompt": row["prompt"], "expected": row["target"], "actual": answer, **trace, **judged})
    criterion = (f"Conversas com histórico gerado pelo modelo após o contexto inicial: igualdade exata após strip, greedy, {response_tokens} tokens. Não é qualidade geral."
                 if any("history" in r for r in rows) else f"Casos documentais: JSON estrito, status e conjunto de evidências corretos; replay: igualdade exata após strip. Greedy, até {response_tokens} tokens. Templates, não documentos reais."
                 if any(r["task"].startswith("grounded_") for r in rows) else
                 f"Igualdade exata após strip; greedy, até {response_tokens} tokens. Validação de templates, não conversa geral.")
    return {"correct": sum(row["exact_match"] for row in outputs), "total": len(outputs), "samples": outputs, "criterion": criterion}


def acceptance(result):
    conversational=[r for r in result["samples"] if r["task"].startswith("conversation_")]
    grounded=[r for r in result["samples"] if r["task"].startswith("grounded_")]
    if conversational and grounded:
        parts={name: acceptance({"samples":samples}) for name,samples in
               (("conversation",conversational),("documents",grounded))}
        return {"passed":all(p["passed"] for p in parts.values()),"objectives":parts,
                "description":"Objetivos separados: cada critério deve passar; nenhuma média entre conversa e documentos.",
                "scope":"Templates controlados; não valida conversa aberta ou projetos reais."}
    if conversational:
        tasks={}
        for row in conversational:
            pair=tasks.setdefault(row["task"],[0,0]);pair[0]+=row["exact_match"];pair[1]+=1
        return {"passed":all(c/t >= (.9 if task.startswith("conversation_context_answer_") else .8) for task,(c,t) in tasks.items()),"counts":tasks,
                "description":"Conversas sintéticas: ao menos 90% nas famílias contextuais e 80% nas demais, com histórico gerado pelo modelo.","scope":"Não valida conversa aberta."}
    grounded = [r for r in result["samples"] if r["task"].startswith("grounded_")]
    if grounded:
        profiles = {}
        for row in grounded:
            counts = profiles.setdefault(row.get("evaluation_profile", "default"), [0, 0])
            counts[0] += row["exact_match"]; counts[1] += 1
        return {"passed": all(r["valid_format"] and r["reference_valid"] for r in grounded) and all(correct/total >= .9 for correct, total in profiles.values()),
                "profiles": profiles,
                "description":"Piloto documental: 100% de formato/referências válidos e ao menos 90% de status e evidências corretos em cada perfil.",
                "scope":"Templates sintéticos; não comprova confiabilidade geral."}
    critical = {task: [0, 0] for task in ("copy", "field", "missing", "insufficient_context")}
    for row in result["samples"]:
        if row["task"] in critical:
            critical[row["task"]][0] += row["exact_match"]
            critical[row["task"]][1] += 1
    passed = all(total > 0 and correct / total >= .8 for correct, total in critical.values())
    return {"passed": passed, "minimum_per_critical_task": .8, "counts": critical,
            "scope": "Critério exploratório para o piloto; não comprova confiabilidade geral."}


def training_indices(train_count,humans,synthetic,batch_size,human_fraction,generator,public=None,public_fraction=0,documents=None,document_fraction=0):
    if not human_fraction and not public_fraction and not document_fraction:
        return torch.randint(train_count,(batch_size,),generator=generator).tolist()
    draw=torch.rand((),generator=generator).item()
    pool=(humans if draw<human_fraction else public if draw<human_fraction+public_fraction
          else documents if draw<human_fraction+public_fraction+document_fraction else synthetic)
    return [pool[i] for i in torch.randint(len(pool),(batch_size,),generator=generator).tolist()]


def validation_losses(model, examples, rows, device, balanced=False):
    if not balanced:
        return {"validation_loss": evaluate(model, examples, device)}
    groups = {name: [example for example, row in zip(examples, rows) if row["task"].startswith(prefix)]
              for name, prefix in (("conversation", "conversation_"), ("documents", "grounded_"))}
    if not all(groups.values()) or sum(map(len, groups.values())) != len(examples):
        raise ValueError("Seleção equilibrada exige somente conversa e documentos, ambos presentes.")
    losses = {name: evaluate(model, items, device) for name, items in groups.items()}
    tokens = {name: sum(label != -100 for _, labels in items for label in labels) for name, items in groups.items()}
    return {"validation_loss": sum(losses[name] * tokens[name] for name in groups) / sum(tokens.values()),
            "selection_loss": sum(losses.values()) / 2,
            **{name + "_validation_loss": loss for name, loss in losses.items()}}


def conversation_indices(pools, batch_size, generator):
    indices = []
    for _ in range(batch_size):
        category = list(pools)[torch.randint(len(pools), (), generator=generator).item()]
        families = pools[category]
        family = list(families)[torch.randint(len(families), (), generator=generator).item()]
        pool = families[family]
        indices.append(pool[torch.randint(len(pool), (), generator=generator).item()])
    return indices


def run(args):
    if args.run_dir.exists() or args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Saída deve ser nova; passos, batch e taxa devem ser positivos.")
    max_seconds=getattr(args,"max_seconds",None)
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0): raise ValueError("Prazo de treino inválido.")
    decision_weight = getattr(args, "document_decision_weight", 1.)
    if not math.isfinite(decision_weight) or decision_weight < 1:
        raise ValueError("Peso documental deve ser finito e >= 1.")
    torch.set_num_threads(4); torch.manual_seed(42)
    device = choose_device(args.device)
    rows, fingerprint = load_instructions(args.data)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    family_sampling = getattr(args, "balanced_conversation_sampling", False)
    semantic_auxiliary=getattr(args,'semantic_auxiliary',False)
    if semantic_auxiliary and not family_sampling:raise ValueError('Supervisão semântica exige conversa equilibrada.')
    first_token_weight=getattr(args,'first_token_weight',1.)
    if not math.isfinite(first_token_weight) or first_token_weight<1 or (first_token_weight>1 and not family_sampling):
        raise ValueError('Peso inicial finito >= 1 exige amostragem equilibrada de conversa.')
    conversation_pools = {"legacy": {}, "context": {}}
    if family_sampling:
        if any(not row["task"].startswith("conversation_") for row in train):
            raise ValueError("Amostragem por família exige somente conversa.")
        for index, row in enumerate(train):
            category = "context" if row["task"].startswith("conversation_context_answer_") else "legacy"
            conversation_pools[category].setdefault(row["task"], []).append(index)
        if not all(conversation_pools.values()): raise ValueError("Exige famílias legadas e contextuais.")
    presence_pools = []
    if getattr(args, "paired_presence_focus", False):
        if not family_sampling or args.batch_size % 2: raise ValueError("Foco pareado exige amostragem de conversa e lote par.")
        grouped = {}
        for i, row in enumerate(train):
            scenario=row.get('scenario',{});text=scenario.get('text','')
            explicit_group=scenario.get('paired_presence_group')
            if row['task']=='conversation_context_answer_support' and (explicit_group is not None or (text.startswith('O projeto não inclui') and ', mas ' in text)):
                grouped.setdefault(('explicit',explicit_group) if explicit_group is not None else ('text',text), []).append(i)
        presence_pools = [pool for pool in grouped.values() if len({train[i]["target"] for i in pool}) > 1]
        if not presence_pools: raise ValueError("Currículo não contém consultas contrastivas sobre o mesmo texto.")
    comparison_pools=[]
    if getattr(args,'paired_facts_focus',False):
        if not family_sampling or args.batch_size%2 or presence_pools:raise ValueError('Comparações pareadas exigem conversa equilibrada, lote par e foco único.')
        grouped={}
        for i,row in enumerate(train):
            key=row.get('scenario',{}).get('comparison_group')
            if key is not None:grouped.setdefault(key,[]).append(i)
        comparison_pools=[pool for pool in grouped.values() if len({train[i]['target'] for i in pool})>1]
        if not comparison_pools:raise ValueError('Nenhuma comparação contrastiva no treino.')
    balanced_selection = getattr(args, "balanced_objective_selection", False)
    generative_selection = getattr(args, "generative_selection", False)
    skip_initial_generation = getattr(args, "skip_initial_generation", False)
    if skip_initial_generation and not generative_selection:
        raise ValueError("Omitir a geração inicial exige seleção gerativa e referência comparativa.")
    selection_patience = getattr(args, "selection_patience", None)
    if selection_patience is not None and (not generative_selection or selection_patience <= 0):
        raise ValueError("Paciência positiva exige seleção gerativa.")
    if generative_selection and balanced_selection:
        raise ValueError("Escolha seleção gerativa ou seleção pela média das losses.")
    if balanced_selection and ({"conversation" if row["task"].startswith("conversation_") else "documents"
                                if row["task"].startswith("grounded_") else "other" for row in validation}
                               != {"conversation", "documents"}):
        raise ValueError("Seleção equilibrada exige validação de conversa e documentos exclusivamente.")
    generation_budget=getattr(args,"response_tokens",64)
    human_fraction=getattr(args,"human_fraction",0.)
    public_fraction=getattr(args,"public_fraction",0.)
    document_fraction=getattr(args,"document_fraction",0.)
    if not (0<=human_fraction<1 and 0<=public_fraction<1 and 0<=document_fraction<1
            and human_fraction+public_fraction+document_fraction<1): raise ValueError("Frações de mistura inválidas.")
    if any(r.get("response_tokens",64)!=generation_budget for r in rows if "history" in r): raise ValueError("Orçamento do corpus deve corresponder ao treino.")
    controlled_validation=[r for r in validation if r["task"] not in {"human_dialogue","public_dialogue"}] if generation_budget>64 else validation
    if generative_selection and (not controlled_validation or any(not r["task"].startswith(("grounded_", "conversation_")) for r in controlled_validation)):
        raise ValueError("Seleção gerativa exige validação controlada por objetivo.")
    source = torch.load(args.init_from, map_location="cpu", weights_only=True)
    if skip_initial_generation and source["settings"].get("objective") != "causal_lm":
        raise ValueError("Omitir a geração inicial é restrito à base de linguagem ainda não ajustada.")
    evaluation_context=getattr(args,"evaluation_context",None) or (256 if generation_budget>64 else source["config"]["block_size"])
    if not 66<=evaluation_context<=source["config"]["block_size"]: raise ValueError("Contexto de avaliação inválido.")
    if source["settings"].get("objective") not in {"causal_lm", "instruction", "document_grounding", "conversation"}:
        raise ValueError("Inicialização exige checkpoint local de linguagem ou instruções.")
    tokenizer = ByteBPE(source["merges"])
    model = GPT(GPTConfig(**source["config"])).to(device)
    model.load_state_dict(source["model"]); model.eval()
    from .finetuning import configure_trainable
    last_blocks = getattr(args, "train_last_blocks", None)
    trainable_count = configure_trainable(model, last_blocks, norm_bias_only=getattr(args, "train_norm_bias_only", False))
    encoded_train = [encode_sample(row, tokenizer, model.config.block_size) for row in train]
    from .decision_loss import decision_weights, weighted_loss, weight_batch, example_mean_loss
    train_weights = ([decision_weights(row, tokenizer, encoded[1], decision_weight)
                      for row, encoded in zip(train, encoded_train)] if decision_weight > 1 else None)
    encoded_val = [encode_sample(row, tokenizer, model.config.block_size) for row in validation]
    human_eval=[encoded_val[i] for i,r in enumerate(validation) if r["task"]=="human_dialogue"]
    evaluation_indices=torch.randperm(len(encoded_train),generator=torch.Generator().manual_seed(43)).tolist()[:128]
    train_eval=[encoded_train[index] for index in evaluation_indices]
    comparison = getattr(args, "compare_with", None)
    if skip_initial_generation and not comparison:
        raise ValueError("Informe uma referência comparativa explicitamente avaliada.")
    baseline_from = getattr(args, "baseline_from", None)
    if baseline_from:
        if comparison: raise ValueError("Cache e comparação alternativa não podem ser usados juntos.")
        from .evaluation_cache import reuse_baseline
        baseline = reuse_baseline(baseline_from, args.init_from, controlled_validation, evaluation_context, generation_budget, device)
    elif comparison:
        previous = torch.load(comparison, map_location="cpu", weights_only=True)
        if evaluation_context>previous["config"]["block_size"]: raise ValueError("Contexto de avaliação não cabe na referência.")
        if any(previous["config"][k]!=source["config"][k] for k in ("vocab_size","width","layers","heads")) or previous["merges"] != source["merges"]:
            raise ValueError("Comparação exige arquitetura e tokenizer iguais.")
        reference=GPT(GPTConfig(**previous["config"])).to(device);reference.load_state_dict(previous["model"]);reference.eval()
        baseline=score(reference,tokenizer,controlled_validation,device,context_limit=evaluation_context, response_tokens=generation_budget)
        del reference
    else: baseline=score(model,tokenizer,controlled_validation,device,context_limit=evaluation_context, response_tokens=generation_budget)
    anchor = None
    anchor_from = getattr(args, "anchor_from", None)
    if anchor_from:
        anchor_weights = torch.load(anchor_from, map_location="cpu", weights_only=True)
        if anchor_weights["config"] != source["config"] or anchor_weights["merges"] != source["merges"]:
            raise ValueError("Âncora exige arquitetura e tokenizer idênticos.")
        anchor = GPT(GPTConfig(**source["config"])).to(device)
        anchor.load_state_dict(anchor_weights["model"]); anchor.eval().requires_grad_(False)
        del anchor_weights
    args.run_dir.mkdir(parents=True)
    objective = "conversation" if any("history" in r for r in rows) else "document_grounding" if any(r["task"].startswith("grounded_") for r in rows) else "instruction"
    settings = {"objective": objective, "seed": 42, "batch_size": args.batch_size, "learning_rate": args.learning_rate,"generation_budget":generation_budget,"human_fraction":human_fraction}
    settings["human_sampling_policy"]="homogeneous_batches"
    settings["trainable_scope"] = "final_normalization_bias" if getattr(args, "train_norm_bias_only", False) else "selected_blocks" if last_blocks is not None else "all_parameters"
    settings["paired_presence_focus"] = bool(presence_pools)
    if presence_pools:settings['paired_presence_groups']=len(presence_pools)
    settings['paired_facts_focus']=bool(comparison_pools)
    settings["training_loss_reduction"] = "example_mean" if family_sampling else "token_mean"
    settings['first_token_weight']=first_token_weight
    settings['semantic_auxiliary']=semantic_auxiliary
    if semantic_auxiliary:
        settings['semantic_auxiliary_initialization']='random_new_training_only_head'
        settings['semantic_auxiliary_weight']=.1
        settings['semantic_auxiliary_learning_rate']=.0003
    if anchor_from:
        settings["anchor_checkpoint"] = str(anchor_from)
        settings["anchor_sha256"] = hashlib.sha256(Path(anchor_from).read_bytes()).hexdigest()
        settings["anchor_policy"] = "kl_only_gold_matching_teacher_tokens_weight_1"
    settings["train_last_blocks"] = last_blocks
    settings["document_decision_weight"] = decision_weight
    settings["checkpoint_selection"] = "objective_mean" if balanced_selection else "token_mean"
    if generative_selection: settings["checkpoint_selection"] = "generated_family_quality"
    settings["selection_patience"] = selection_patience
    settings["initial_candidate_generation_measured"] = not skip_initial_generation
    settings["conversation_sampling"] = "equal_legacy_context_then_uniform_family_per_example" if family_sampling else "uniform_examples"
    settings["public_fraction"]=public_fraction
    settings["document_fraction"]=document_fraction
    schemas = {r.get("grounding_schema", 1) for r in rows if r["task"].startswith("grounded_")}
    if len(schemas) > 1:
        raise ValueError("Use um único contrato documental por treino.")
    if schemas:
        settings["grounding_schema"] = next(iter(schemas))
    info = {"objective": objective, "initialization": "local_checkpoint_new_optimizer", "resume_from": str(args.init_from),
            "base_step": source["step"], "source_objective": source["settings"]["objective"], "config": asdict(model.config), "parameters": sum(p.numel() for p in model.parameters()),
            "baseline_checkpoint": str(comparison or args.init_from), "train_evaluation_examples": len(train_eval),
            "paired_evaluation_context":evaluation_context,"paired_evaluation_response_tokens":generation_budget,
            "human_validation_loss_examples":sum(r["task"]=="human_dialogue" for r in validation),
            "train_evaluation_policy":"amostra fixa sem reposição, seed 43",
            "data": str(args.data), "dataset_fingerprint": fingerprint, "settings": settings,
            "document_splits": dict(Counter(row["split"] for row in rows)), "train_ids": [r["id"] for r in train],
            "validation_ids": [r["id"] for r in validation], "test_examples_used": 0,
            "device": device, "torch": str(torch.__version__), "python": platform.python_version()}
    from .evaluation_cache import protocol_fingerprint
    info["evaluation_protocol_hash"] = protocol_fingerprint()
    info["trainable_parameters"] = trainable_count
    if baseline_from: info["baseline_evaluation_reused_from"] = str(baseline_from)
    info["mps_batch_padding_multiple"] = 128 if device == "mps" else None
    info["mps_generation_padding_multiple"] = 128 if device == "mps" else None
    (args.run_dir/"run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    semantic_head=torch.nn.Linear(model.config.width,6).to(device) if semantic_auxiliary else None
    model_parameters=[p for p in model.parameters() if p.requires_grad]
    parameters=model_parameters+(list(semantic_head.parameters()) if semantic_head is not None else [])
    groups=([dict(params=model_parameters),dict(params=semantic_head.parameters(),lr=.0003)] if semantic_head is not None else parameters)
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=.01)
    generator = torch.Generator().manual_seed(42)
    def checkpoint(filename,step):
        path=args.run_dir/filename
        save_checkpoint(path,model,optimizer,tokenizer,generator,step,fingerprint,settings)
        if semantic_head is not None:
            target=path.with_suffix('.auxiliary.pt');temporary=target.with_suffix('.tmp')
            torch.save(dict(step=step,head={k:v.detach().cpu() for k,v in semantic_head.state_dict().items()},
                            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_only=True),temporary)
            temporary.replace(target)
    humans=[i for i,r in enumerate(train) if r["task"]=="human_dialogue"]
    public=[i for i,r in enumerate(train) if r["task"]=="public_dialogue"]
    documents=[i for i,r in enumerate(train) if r["task"].startswith("grounded_")]
    synthetic=[i for i,r in enumerate(train) if r["task"] not in {"human_dialogue","public_dialogue"}
               and not (document_fraction and r["task"].startswith("grounded_"))]
    if human_fraction and (not humans or not synthetic): raise ValueError("Mistura exige humanos e sintéticos.")
    if public_fraction and (not public or not synthetic): raise ValueError("Mistura pública exige fonte pública e sintéticos.")
    if public and not public_fraction: raise ValueError("Defina --public-fraction explicitamente para a fonte pública.")
    if document_fraction and (not documents or not synthetic):raise ValueError("Mistura documental exige documentos e exemplos autorais não documentais.")
    started, best, processed, memory = time.perf_counter(), float("inf"), 0, 0
    best_quality = best_generation = None
    stale_evaluations = 0
    selection_plateau = False
    metrics = []
    driver_memory = 0
    human_examples_processed=0
    human_tokens_processed=0
    public_examples_processed=0
    public_tokens_processed=0
    document_examples_processed=0
    document_tokens_processed=0
    def record(step):
        nonlocal best, best_quality, best_generation, stale_evaluations
        item = {"step": step, "train_loss": evaluate(model, train_eval, device),
                **validation_losses(model, encoded_val, validation, device, balanced_selection),
                "elapsed_seconds": time.perf_counter()-started}
        if device == "mps":
            item["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            item["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        if human_eval:item["human_validation_loss"]=evaluate(model,human_eval,device)
        metrics.append(item)
        with (args.run_dir/"metrics.jsonl").open("a") as f: f.write(json.dumps(item)+"\n")
        print(json.dumps(item), flush=True)
        checkpoint('last.pt',step)
        if step == 0 and skip_initial_generation:
            print("Base inicial não avaliada por geração e excluída da seleção; referência comparativa medida separadamente.", flush=True)
            return
        if generative_selection:
            from .generative_selection import quality_key
            generation = (baseline if step == 0 and not comparison else
                          score(model, tokenizer, controlled_validation, device, evaluation_context, generation_budget))
            quality = quality_key(generation)
            (args.run_dir / f"generation-{step}.json").write_text(json.dumps(dict(quality=quality, result=generation), ensure_ascii=False, indent=2))
            print(json.dumps(dict(generation_step=step, quality=quality, correct=generation["correct"], total=generation["total"])), flush=True)
            if best_quality is None or quality > best_quality:
                best_quality, best_generation = quality, generation
                stale_evaluations = 0
                checkpoint('best.pt',step)
            else:
                stale_evaluations += 1
            return
        selection_loss = item.get("selection_loss", item["validation_loss"])
        if selection_loss < best:
            best = selection_loss
            checkpoint('best.pt',step)
    record(0)
    for step in range(1, args.steps+1):
        model.train()
        if (presence_pools or comparison_pools) and torch.rand((), generator=generator).item() < .5:
            from .evidence_classifier import paired_indices
            indices = paired_indices(train, presence_pools or comparison_pools, args.batch_size, generator, label_key="target")
        else:
            indices=(conversation_indices(conversation_pools, args.batch_size, generator) if family_sampling else
                     training_indices(len(train),humans,synthetic,args.batch_size,human_fraction,generator,public,public_fraction,documents,document_fraction))
        x,y = batch(encoded_train,indices,device,model.config.block_size)
        optimizer.zero_grad(set_to_none=True)
        if semantic_head is not None:
            from .semantic_auxiliary import semantic_class,semantic_forward
            classes=torch.tensor([semantic_class(train[i]) for i in indices],device=device)
            logits,semantic_loss=semantic_forward(model,x,y,semantic_head,classes)
        else:logits = model(x)
        loss = (weighted_loss(logits, y, weight_batch(train_weights, indices, y.shape, device))
                if train_weights is not None else example_mean_loss(logits, y,first_token_weight) if family_sampling else F.cross_entropy(logits.flatten(0,1), y.flatten(), ignore_index=-100))
        if semantic_head is not None:loss=loss+.1*semantic_loss
        if anchor is not None:
            from .distillation import correct_teacher_kl
            with torch.no_grad(): anchor_logits = anchor(x)
            loss = loss + correct_teacher_kl(logits, anchor_logits, y)
        if not torch.isfinite(loss).item(): raise ValueError("Loss não finita; checkpoint anterior preservado.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True); optimizer.step()
        processed += (y != -100).sum().item()
        human_examples_processed+=sum(train[i]["task"]=="human_dialogue" for i in indices)
        for j,i in enumerate(indices):
            if train[i]["task"]=="human_dialogue":human_tokens_processed+=(y[j]!=-100).sum().item()
            if train[i]["task"]=="public_dialogue":
                public_examples_processed+=1;public_tokens_processed+=(y[j]!=-100).sum().item()
            if train[i]["task"].startswith("grounded_"):
                document_examples_processed+=1;document_tokens_processed+=(y[j]!=-100).sum().item()
        if device == "mps":
            memory = max(memory, torch.mps.current_allocated_memory())
            driver_memory = max(driver_memory, torch.mps.driver_allocated_memory())
        timed_out=max_seconds is not None and time.perf_counter()-started>=max_seconds
        if step % args.eval_every == 0 or step == args.steps or timed_out: record(step)
        if selection_patience is not None and stale_evaluations >= selection_patience:
            selection_plateau = True
            break
        if timed_out: break
    summary = {"start_step": 0, "final_step": step,"stop_reason":"selection_plateau" if selection_plateau else "time_limit" if timed_out else "steps", "elapsed_seconds": time.perf_counter()-started,
               "tokens_processed": processed, "initial": metrics[0], "final": metrics[-1],
               "human_examples_processed":human_examples_processed,"human_tokens_processed":human_tokens_processed,
               "public_examples_processed":public_examples_processed,"public_tokens_processed":public_tokens_processed,
               "document_examples_processed":document_examples_processed,"document_tokens_processed":document_tokens_processed,
               "mps_allocated_bytes_max_observed": memory, "mps_driver_allocated_bytes_max_observed": driver_memory, "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024)}
    (args.run_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    if generative_selection and best_generation is None:
        raise ValueError("Nenhum candidato ajustado teve avaliação gerativa completa.")
    selected = torch.load(args.run_dir/"best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"]); model.eval()
    result = {"baseline": baseline, "selected": best_generation if generative_selection else score(model, tokenizer, controlled_validation, device,context_limit=evaluation_context, response_tokens=generation_budget), "selected_step": selected["step"], "test_examples_used": 0}
    result["acceptance"] = acceptance(result["selected"])
    (args.run_dir/"instruction-evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if objective=="conversation":
        samples=[{"prompt":r["prompt"],"continuation":r["actual"]} for r in result["selected"]["samples"][:3]]
        (args.run_dir/"report-samples.json").write_text(json.dumps(samples,ensure_ascii=False,indent=2))
    else:
        save_samples(args.run_dir, model, tokenizer, device, [(row["prompt"], [tokenizer.BOS]+tokenizer.encode(row["prompt"])+[tokenizer.SEP]) for row in validation[:3]])
    print(f"Relatório: {create_report(args.run_dir)}", flush=True)
    print(f'Acertos exatos: {baseline["correct"]}/{baseline["total"]} -> {result["selected"]["correct"]}/{result["selected"]["total"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ai/data/authored/instructions.jsonl"))
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--baseline-from", type=Path, help="Reusar geração já auditada apenas com pesos, protocolo e validação idênticos.")
    parser.add_argument("--anchor-from", type=Path, help="Preservar probabilidades corretas de checkpoint próprio; tokens divergentes do alvo não são ancorados.")
    parser.add_argument("--compare-with", type=Path, help="Checkpoint local anterior avaliado nos mesmos pedidos novos.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--response-tokens",type=int,default=64)
    parser.add_argument("--train-norm-bias-only", action="store_true", help="Ajustar somente o viés da normalização final, preservando blocos, embeddings e projeção.")
    parser.add_argument("--train-last-blocks", type=int, help="Congelar embeddings e blocos iniciais; treinar apenas últimos N blocos, norm e saída.")
    parser.add_argument("--document-decision-weight", type=float, default=1., help="Peso dos valores status/unit no treino; avaliação continua sem pesos.")
    parser.add_argument("--balanced-objective-selection", action="store_true", help="Selecionar checkpoint pela média das losses de conversa e documentos (50% cada).")
    parser.add_argument("--generative-selection", action="store_true", help="Selecionar por respostas geradas e pior família; loss continua registrada. Desenvolvimento, não teste cego.")
    parser.add_argument("--selection-patience", type=int, help="Encerrar após N avaliações gerativas consecutivas sem ganho.")
    parser.add_argument("--paired-presence-focus", action="store_true", help="Metade dos lotes compara consultas de presença com respostas diferentes no mesmo texto; restante usa replay equilibrado.")
    parser.add_argument('--paired-facts-focus',action='store_true',help='Metade dos lotes compara repetição e contradição dos mesmos valores de treino; restante usa replay equilibrado.')
    parser.add_argument("--balanced-conversation-sampling", action="store_true", help="Lotes com 50% de famílias legadas e 50% contextuais; famílias equiprováveis em cada grupo.")
    parser.add_argument('--first-token-weight',type=float,default=1.,help='Peso inicial por resposta no treino equilibrado; avaliação completa continua sem pesos.')
    parser.add_argument('--semantic-auxiliary',action='store_true',help='Classificar o tipo de resposta no treino; cabeça auxiliar não participa da inferência.')
    parser.add_argument("--skip-initial-generation", action="store_true", help="Excluir a base causal não ajustada da seleção; exige referência externa e seleção gerativa.")
    parser.add_argument("--human-fraction",type=float,default=0.)
    parser.add_argument("--public-fraction",type=float,default=0.)
    parser.add_argument("--document-fraction",type=float,default=0.,help="Fração de lotes documentais separados dos demais exemplos autorais.")
    parser.add_argument("--max-seconds",type=float)
    parser.add_argument("--evaluation-context",type=int,help="Janela comum explícita para avaliação de conversas completas.")
    args = parser.parse_args()
    existed=args.run_dir.exists()
    try: run(args)
    except (ValueError, OSError, RuntimeError) as error:
        failure_report=None
        if not existed:
            try:failure_report=record_failure(args.run_dir,error)
            except (ValueError,OSError,RuntimeError) as report_error:
                print(f"Falha ao gerar relatório da interrupção: {report_error}",flush=True)
        parser.exit(1, f"Erro: {error}\n"+(f"Relatório da interrupção: {failure_report}\n" if failure_report else ""))


if __name__ == "__main__": main()
