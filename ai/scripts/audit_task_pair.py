"""Recompõe a coorte original sem esconder regressões entre ajustes especializados."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_context_answers import audit
from confere_ai.evaluation_cache import protocol_fingerprint, reuse_baseline
from confere_ai.instruct import load_instructions


def combine(baseline, parts):
    by_id = {}
    for part in parts:
        if part["total"] != len(part["samples"]) or part["correct"] != sum(r["exact_match"] for r in part["samples"]):
            raise ValueError("Totais não correspondem às amostras.")
        for row in part["samples"]:
            if row["id"] in by_id: raise ValueError("Caso duplicado entre ajustes.")
            by_id[row["id"]] = row
    if set(by_id) != {r["id"] for r in baseline["samples"]}:
        raise ValueError("Coorte conjunta incompleta ou com casos extras.")
    selected = [by_id[r["id"]] for r in baseline["samples"]]
    if [(r["id"], r["task"], r["prompt"], r["expected"]) for r in selected] != [
            (r["id"], r["task"], r["prompt"], r["expected"]) for r in baseline["samples"]]:
        raise ValueError("Casos ou alvos foram alterados.")
    return dict(correct=sum(r["exact_match"] for r in selected), total=len(selected), samples=selected,
                criterion="Mesma coorte original; cada objetivo usa seu ajuste especializado. Desenvolvimento, sem teste cego.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "documents", "conversation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    rows, digest = load_instructions(args.source)
    validation = [r for r in rows if r["split"] == "validation"]
    baseline = reuse_baseline(args.baseline, args.baseline / "best.pt", validation, 1024, 256)
    parts = []; checkpoints = {}
    for name, prefix in (("documents", "grounded_"), ("conversation", "conversation_")):
        directory = getattr(args, name)
        info = json.loads((directory / "run.json").read_text())
        subset, subset_digest = load_instructions(Path(info["data"]))
        if subset != [r for r in rows if r["task"].startswith(prefix)] or subset_digest != info["dataset_fingerprint"]:
            raise ValueError("Currículo especializado difere da partição original.")
        if (info["evaluation_protocol_hash"] != protocol_fingerprint() or info["device"] != "mps"
            or info["paired_evaluation_context"] != 1024 or info["paired_evaluation_response_tokens"] != 256):
            raise ValueError("Protocolo especializado difere da referência.")
        result = json.loads((directory / "instruction-evaluation.json").read_text())
        if result["test_examples_used"] != 0: raise ValueError("Teste reservado foi usado.")
        if [(r["id"], r["task"], r["prompt"], r["expected"]) for r in result["selected"]["samples"]] != [
                (r["id"], r["task"], r["prompt"], r["target"]) for r in subset if r["split"] == "validation"]:
            raise ValueError("Geração não corresponde à validação do objetivo declarado.")
        checkpoint_path = directory / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["step"] != result["selected_step"]: raise ValueError("Passo dos pesos difere da geração.")
        parts.append(result["selected"])
        checkpoints[name] = dict(path=str(checkpoint_path), step=checkpoint["step"], sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    result = dict(baseline=baseline, selected=combine(baseline, parts), test_examples_used=0)
    result["audit"] = audit(result)
    result["checkpoints"] = checkpoints; result["source_sha256"] = digest
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(correct=result["selected"]["correct"], total=result["selected"]["total"], audit=result["audit"]), ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
