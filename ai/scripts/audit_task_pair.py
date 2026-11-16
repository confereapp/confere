"""Recompõe a coorte original sem esconder regressões entre ajustes especializados."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_context_answers import audit
from confere_ai.evaluation_cache import protocol_fingerprint, reuse_baseline
from confere_ai.instruct import load_instructions
from confere_ai.generative_selection import quality_key


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



def original_partition(original, subset):
    reserved = [r for r in original if r["split"] != "train"]
    ids = {r["id"] for r in reserved}
    actual = [r for r in subset if r["split"] != "train"]
    extras = [r for r in actual if r["id"] not in ids]
    if ([r for r in actual if r["id"] in ids] != reserved
            or any(r["split"] != "validation" or not r["id"].startswith("development-")
                   or not r["task"].startswith(("grounded_", "conversation_")) for r in extras)):
        raise ValueError("Partição original alterada ou exemplos extras não declarados como desenvolvimento.")
    return extras

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
        original = [r for r in rows if r["task"].startswith(prefix)]
        extras = original_partition(original, subset)
        if subset_digest != info["dataset_fingerprint"]:
            raise ValueError("Hash do currículo especializado difere da execução.")
        checkpoint_path = directory / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["settings"].get("objective") == "document_classifier":
            if name != "documents" or info["test_examples_used"] != 0:
                raise ValueError("Classificador não pode substituir conversa ou usar teste reservado.")
            from confere_ai.evidence_classifier import EvidencePredictor, examples
            from train_evidence_classifier import score as classifier_score
            predictor = EvidencePredictor(checkpoint, "mps")
            if predictor.config.block_size != 1024: raise ValueError("Contexto especializado incompatível.")
            validation_rows = [r for r in subset if r["split"] == "validation"]
            encoded = examples(validation_rows, predictor.tokenizer, predictor.config.block_size, readout=predictor.readout)
            result = dict(selected=classifier_score(predictor.model, encoded, validation_rows, "mps", predictor.thresholds),
                          selected_step=checkpoint["step"], test_examples_used=0)
            del predictor
        else:
            if (info["evaluation_protocol_hash"] != protocol_fingerprint() or info["device"] != "mps"
                or info["paired_evaluation_context"] != 1024 or info["paired_evaluation_response_tokens"] != 256):
                raise ValueError("Protocolo especializado difere da referência.")
            result = json.loads((directory / "instruction-evaluation.json").read_text())
        if result["test_examples_used"] != 0: raise ValueError("Teste reservado foi usado.")
        if [(r["id"], r["task"], r["prompt"], r["expected"]) for r in result["selected"]["samples"]] != [
                (r["id"], r["task"], r["prompt"], r["target"]) for r in subset if r["split"] == "validation"]:
            raise ValueError("Geração não corresponde à validação do objetivo declarado.")
        if checkpoint["step"] != result["selected_step"]: raise ValueError("Passo dos pesos difere da geração.")
        original_ids = {r["id"] for r in original if r["split"] == "validation"}
        original_samples = [r for r in result["selected"]["samples"] if r["id"] in original_ids]
        parts.append(dict(samples=original_samples, total=len(original_samples), correct=sum(r["exact_match"] for r in original_samples)))
        checkpoints[name] = dict(path=str(checkpoint_path), step=checkpoint["step"], sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                                 additional_development_validation=len(extras), prediction_type=checkpoint["settings"].get("objective"), quality=quality_key(result["selected"]))
    result = dict(baseline=baseline, selected=combine(baseline, parts), test_examples_used=0)
    result["audit"] = audit(result)
    result["audit"]["specialist_quality_passed"] = all(c["quality"][0] for c in checkpoints.values())
    result["audit"]["controlled_criteria_pass"] &= result["audit"]["specialist_quality_passed"]
    result["checkpoints"] = checkpoints; result["source_sha256"] = digest
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(correct=result["selected"]["correct"], total=result["selected"]["total"], audit=result["audit"]), ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
