"""Ablação offline: repetir a pergunta após as unidades sem remover a fonte."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.evaluation_cache import reuse_baseline
from confere_ai.grounding import judge, prompt_for
from confere_ai.instruct import load_instructions
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


def repeated_prompt(row):
    original = prompt_for(row["requirement"], row["units"], row.get("grounding_schema", 1))
    if original != row["prompt"]:
        raise ValueError("Prompt original não corresponde às unidades declaradas.")
    return original + "\nRequisito a conferir: " + row["requirement"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    info = json.loads((args.run_dir / "run.json").read_text())
    rows, digest = load_instructions(Path(info["data"]))
    assert digest == info["dataset_fingerprint"]
    validation = [r for r in rows if r["split"] == "validation"]
    checkpoint_path = args.run_dir / "best.pt"
    baseline = reuse_baseline(args.run_dir, checkpoint_path, validation, 1024, 256)
    reference = {r["id"]: r for r in baseline["samples"]}
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"])).to("mps")
    model.load_state_dict(checkpoint["model"]); model.eval()
    tokenizer = ByteBPE(checkpoint["merges"])
    results = []
    for row in validation:
        if not row["task"].startswith("grounded_"): continue
        prompt = repeated_prompt(row)
        prefix = [tokenizer.BOS] + tokenizer.encode(prompt) + [tokenizer.SEP]
        budget = min(256, model.config.block_size - len(prefix))
        if len(tokenizer.encode(row["target"])) + 1 > budget:
            raise ValueError("A repetição não deixa espaço para o alvo inteiro; ensaio recusado.")
        output = model.generate(torch.tensor([prefix], device="mps"), budget, tokenizer.EOS)
        actual = tokenizer.decode(output[0, len(prefix):].tolist())
        results.append(dict(id=row["id"], profile=row.get("evaluation_profile"), prompt=prompt,
                            expected=row["target"], before=reference[row["id"]]["actual"], actual=actual,
                            before_correct=reference[row["id"]]["exact_match"], response_budget=budget, **judge(actual, row)))
    profiles = {}
    for row in results:
        counts = profiles.setdefault(row["profile"], [0, 0, 0])
        counts[0] += row["before_correct"]; counts[1] += row["exact_match"]; counts[2] += 1
    result = dict(checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  before_correct=sum(r["before_correct"] for r in results), correct=sum(r["exact_match"] for r in results),
                  total=len(results), profiles=profiles, samples=results, adopted=False,
                  note="Somente desenvolvimento offline; mesmo texto/unidades e alvos, requisito repetido ao final. Não altera a API nem comprova qualidade fora da coorte.")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__": main()
