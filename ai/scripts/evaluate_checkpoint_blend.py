"""Ensaio offline de média igual entre dois ajustes da mesma rede local."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from confere_ai.evaluation_cache import protocol_fingerprint, reuse_baseline
from confere_ai.instruct import acceptance, load_instructions, score
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE
from confere_ai.train import save_checkpoint


def blend_states(left, right):
    if left.keys() != right.keys():
        raise ValueError("Tensores de origem incompatíveis.")
    result = {}
    for name, first in left.items():
        second = right[name]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("Formato de tensor incompatível: " + name)
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("Tensor de origem não finito: " + name)
        if first.is_floating_point():
            result[name] = torch.lerp(first, second, .5)
            if not torch.isfinite(result[name]).all():
                raise ValueError("Média não finita: " + name)
        elif torch.equal(first, second):
            result[name] = first.clone()
        else:
            raise ValueError("Buffer não flutuante divergente: " + name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "baseline-run", "data", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists(): parser.error("Saída deve ser nova.")
    torch.set_num_threads(4); torch.manual_seed(42)
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    rows, digest = load_instructions(args.data)
    for key in ("format_version", "config", "merges", "dataset_fingerprint"):
        if left[key] != right[key]: parser.error("Origens incompatíveis: " + key)
    if left["dataset_fingerprint"] != digest: parser.error("Corpus difere das origens.")
    for key in ("objective", "grounding_schema", "generation_budget"):
        if left["settings"].get(key) != right["settings"].get(key): parser.error("Contrato incompatível: " + key)
    if left["settings"].get("grounding_schema") != 2 or left["settings"].get("generation_budget") != 256:
        parser.error("Este ensaio exige contrato V2 e orçamento 256.")
    validation = [r for r in rows if r["split"] == "validation"]
    context = left["config"]["block_size"]
    baseline = reuse_baseline(args.baseline_run, args.left, validation, context, 256)
    model = GPT(GPTConfig(**left["config"]))
    model.load_state_dict(blend_states(left["model"], right["model"]))
    tokenizer = ByteBPE(left["merges"])
    origins = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), weight=.5)
               for path in (args.left, args.right)]
    settings = dict(left["settings"], train_last_blocks=None, checkpoint_selection="fixed_equal_interpolation", blend_sources=origins)
    args.run_dir.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    save_checkpoint(args.run_dir / "best.pt", model, optimizer, tokenizer, torch.Generator().manual_seed(42), 0, digest, settings)
    info = dict(objective=settings["objective"], method="equal_parameter_interpolation_no_training", sources=origins,
                optimizer_reset=True, data=str(args.data), dataset_fingerprint=digest, config=asdict(model.config), device="mps",
                paired_evaluation_context=context, paired_evaluation_response_tokens=256, evaluation_protocol_hash=protocol_fingerprint(), settings=settings)
    (args.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    model = model.to("mps"); model.eval()
    result = dict(baseline=baseline, selected=score(model, tokenizer, validation, "mps", context, 256), selected_step=0, test_examples_used=0)
    result["acceptance"] = acceptance(result["selected"])
    (args.run_dir / "instruction-evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(before=baseline["correct"], after=result["selected"]["correct"], total=len(validation), adopted=False)), flush=True)


if __name__ == "__main__": main()
