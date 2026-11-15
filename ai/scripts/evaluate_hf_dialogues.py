"""Comparação de geração e loss nas validações públicas; não é nota de conversa geral."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.instruct import load_instructions, score
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE
from confere_ai.train import evaluate
from confere_ai.training_data import encode_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    rows, digest = load_instructions(args.data)
    rows = [r for r in rows if r["split"] == "validation" and r["task"] in {"human_dialogue", "public_dialogue"}]
    assert rows and all(r["split"] != "test" for r in rows)
    result = dict(dataset_sha256=digest, count=len(rows), test_examples_used=0, adopted=False,
                  note="Loss teacher-forced e geração greedy; validações públicas não comprovam conversa geral. Linhagem de exposição humana não auditada.")
    torch.set_num_threads(4)
    for name, path in [("before", args.before), ("after", args.after)]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        tokenizer = ByteBPE(checkpoint["merges"])
        model = GPT(GPTConfig(**checkpoint["config"])).to("mps")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        encoded = [encode_sample(row, tokenizer, model.config.block_size) for row in rows]
        result[name] = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            step=checkpoint["step"], loss=evaluate(model, encoded, "mps"),
                            generation=score(model, tokenizer, rows, "mps", context_limit=1024, response_tokens=256))
        del model
        torch.mps.empty_cache()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: {"loss":result[k]["loss"],"exact":result[k]["generation"]["correct"]} for k in ["before","after"]}), flush=True)


if __name__ == "__main__": main()
