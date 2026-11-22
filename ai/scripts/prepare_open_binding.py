"""Associação entre requisito e fonte sem depender de prefixo lexical fixo."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import torch
from confere_ai.grounding import prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample
from prepare_lexical_binding import documents, expected


def build(source, count=6000):
    removed = [r["id"] for r in source if r["split"] == "train" and "a especificação prevê" in r["prompt"].casefold()]
    removed_set = set(removed)
    rows = [r for r in source if r["id"] not in removed_set]
    known = {r["prompt"].strip().casefold(): json.loads(r["target"]) for r in rows}
    rng = random.Random(161117)
    prepared = {}
    for row in documents(count, feature_prefix=""):
        row["id"] = "open-" + row["id"]; row["project_id"] = "open-" + row["project_id"]
        row["document"] = "open-" + row["document"]
        if row["document"] in prepared:
            units, padding = prepared[row["document"]]
            row["units"] = copy.deepcopy(units)
        else:
            padding = []
            for unit in row["units"]:
                repeats = rng.randrange(5); padding.append(repeats)
                filler = " A equipe registrou as atividades e distribuiu tarefas." * repeats
                unit["text"] = (filler.strip() + " " + unit["text"]).strip() if rng.random() < .5 else unit["text"] + filler
            prepared[row["document"]] = (copy.deepcopy(row["units"]), padding)
        row["prompt"] = prompt_for(row["requirement"], row["units"], 2)
        row["scenario"]["padding"] = padding
        assert json.loads(row["target"]) == expected(row["scenario"]["units"], row["scenario"]["feature"])
        key = row["prompt"].strip().casefold()
        if key in known:
            if known[key] != json.loads(row["target"]): raise ValueError("Alvos incompatíveis para o mesmo contexto.")
            continue
        known[key] = json.loads(row["target"]); rows.append(row)
    assert [r for r in rows if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    return rows, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source); rows, removed = build(source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    for row in rows: encode_sample(row, tokenizer, checkpoint["config"]["block_size"])
    args.output.mkdir(parents=True); path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    splits=dict(Counter(r["split"] for r in rows)), removed_ambiguous_train_ids=removed,
                    note="Originais preservados. Somente treino alterado: remove afirmações de planejamento ambíguas, inclui termos inventados sem prefixo fixo e distratores. Validação original intacta, sem teste cego.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); print(json.dumps({k:v for k,v in manifest.items() if k!='removed_ambiguous_train_ids'},ensure_ascii=False))


if __name__ == "__main__": main()
