"""Piloto curto de conversas públicas com replay documental e fontes rastreáveis."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from confere_ai.instruct import load_instructions
from confere_ai.tokenizer import ByteBPE
from confere_ai.training_data import encode_sample

SOURCE = "https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft"


def public_rows(messages, revision):
    # Não descarte instruções de sistema: este formato de treino ainda não as representa.
    if (not messages or len(messages) % 2 or any(
        m.get("role") != ("user" if i % 2 == 0 else "assistant")
        or not isinstance(m.get("content"), str) or not m["content"].strip()
        for i, m in enumerate(messages))):
        return []
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    bucket = int(digest, 16) % 100
    split = "test" if bucket < 5 else "validation" if bucket < 10 else "train"
    return [dict(id=f"hf-{digest}-{i}", project_id="hf-" + digest, split=split,
                 task="public_dialogue", history=[[m["content"], messages[j+1]["content"]]
                 for j, m in enumerate(messages[:i-1]) if j % 2 == 0],
                 prompt=messages[i-1]["content"], target=messages[i]["content"],
                 turn_index=0, response_tokens=256,
                 provenance=dict(kind="public_model_generated", source=SOURCE, revision=revision,
                                 license="Apache-2.0", note="Sintético; sem revisão humana local."))
            for i in range(1, len(messages), 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--humans", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    metadata = json.loads((args.source / "source.json").read_text())
    assert metadata["source"] == SOURCE and metadata["license"] == "Apache-2.0"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(ckpt["merges"])
    rows, base_hash = load_instructions(args.base)
    seen = {json.dumps([r.get("history", []), r["prompt"].strip().casefold()], ensure_ascii=False) for r in rows}
    counts = Counter()
    reserved = []
    def add(row):
        if any(name in row["target"].casefold() for name in ("open assistant", "openassistant", "laion", "chatgpt", "openai", "qwen", "aira")):
            counts["other_assistant_identity"] += 1
            return
        key = json.dumps([row["history"], row["prompt"].strip().casefold()], ensure_ascii=False)
        if key in seen: counts["duplicate"] += 1; return
        try: encode_sample(row, tokenizer, ckpt["config"]["block_size"])
        except ValueError: counts["too_long"] += 1; return
        seen.add(key)
        if row["split"] == "test": reserved.append(row)
        else: rows.append(row)
        counts[row["task"] + "_" + row["split"]] += 1
    humans, human_hash = load_instructions(args.humans)
    for row in humans:
        if row["task"] == "human_dialogue": add(row)
    for name, digest in sorted(metadata["files"].items()):
        raw = (args.source / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        for item in json.loads(raw)["rows"]:
            value = item["row"]
            if value["instruct_score"] < 4: counts["low_score"] += 1; continue
            candidates = public_rows(value["messages"], metadata["revision"])
            if not candidates: counts["unsupported_roles"] += 1
            for row in candidates: add(row)
    assert counts["public_dialogue_train"] and counts["human_dialogue_train"]
    for row in rows: encode_sample(row, tokenizer, ckpt["config"]["block_size"])
    args.output.mkdir(parents=True)
    target = args.output / "examples.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    load_instructions(target)
    (args.output / "reserved-test.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reserved))
    manifest = dict(base_sha256=base_hash, humans_sha256=human_hash, source=SOURCE, source_snapshot=metadata,
                    counts=dict(counts), splits=dict(Counter(r["split"] for r in rows)),
                    dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    note="Amostra inicial não aleatória; validação pública por conversa, sem avaliação cega da linhagem histórica. Teste reservado não é usado no treino.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
