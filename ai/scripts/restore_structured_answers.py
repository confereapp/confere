"""Restaura formatos autorais históricos de treino sem alterar a validação."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from confere_ai.instruct import load_instructions
from prepare_context_answers import answer


def restore(source, histories):
    result = list(source); added = 0
    for index, history in enumerate(histories):
        for original in history:
            if original["split"] != "train" or original["task"] not in {
                    "conversation_context_answer_fact", "conversation_context_answer_missing", "conversation_context_answer_conflict"}:
                continue
            row = copy.deepcopy(original)
            assert row["target"] == answer(row["scenario"]["facts"], row["scenario"]["field"])
            row["id"] = f"restored-{index}-" + row["id"]
            row["project_id"] = f"restored-{index}-" + row["project_id"]
            result.append(row); added += 1
    assert [r for r in result if r["split"] != "train"] == [r for r in source if r["split"] != "train"]
    return result, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--historical", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    source, digest = load_instructions(args.source)
    loaded = [load_instructions(path) for path in args.historical]
    rows, added = restore(source, [r for r, _ in loaded])
    args.output.mkdir(parents=True); path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); load_instructions(path)
    manifest = dict(source_sha256=digest, historical_sha256=[d for _, d in loaded], added=added,
                    dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), splits=dict(Counter(r["split"] for r in rows)),
                    note="Somente restauração de exemplos históricos de treino. Nenhum alvo de validação/teste copiado; validação original preservada.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
