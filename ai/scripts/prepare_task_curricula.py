"""Separa objetivos sem remover casos nem alterar alvos de validação."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from confere_ai.instruct import load_instructions


def partition(rows):
    result = {"documents": [], "conversation": []}
    for row in rows:
        if row["task"].startswith("grounded_"): result["documents"].append(row)
        elif row["task"].startswith("conversation_"): result["conversation"].append(row)
        else: raise ValueError("Corpus deve conter apenas os dois objetivos controlados.")
    assert sum(map(len, result.values())) == len(rows)
    for subset in result.values():
        if not all(any(row["split"] == split for row in subset) for split in ("train", "validation")):
            raise ValueError("Cada objetivo exige treino e validação.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    rows, digest = load_instructions(args.source)
    subsets = partition(rows)
    args.output.mkdir(parents=True)
    manifest = dict(source_sha256=digest, objectives={}, note="Separação de treino, sem novos alvos ou mudança da validação; a aprovação conjunta continua exigindo ambos os objetivos.")
    for name, subset in subsets.items():
        folder = args.output / name; folder.mkdir()
        path = folder / "examples.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset))
        _, fingerprint = load_instructions(path)
        manifest["objectives"][name] = dict(dataset_sha256=fingerprint, splits=dict(Counter(r["split"] for r in subset)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
