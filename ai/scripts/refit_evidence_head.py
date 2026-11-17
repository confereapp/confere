"""Reajuste convexo na CPU: BCE sem pesos e normalização derivada só do treino."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.evidence_classifier import EvidenceClassifier, fold_normalization
from confere_ai.generative_selection import quality_key
from confere_ai.instruct import load_instructions, acceptance
from confere_ai.model import GPT, GPTConfig
from probe_evidence_head import evaluate_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists(): parser.error("Saída deve ser nova.")
    torch.set_num_threads(4); torch.manual_seed(42)
    info = json.loads((args.source / "run.json").read_text())
    if info.get("readout") != "shared_json_decision_prefix_v1": parser.error("Representação incompatível.")
    rows, digest = load_instructions(Path(info["data"]))
    if digest != info["dataset_fingerprint"]: parser.error("Corpus foi alterado.")
    validation = [r for r in rows if r["split"] == "validation"]
    cache = torch.load(args.source / "features.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.source / "best.pt", map_location="cpu", weights_only=True)
    model = EvidenceClassifier(GPT(GPTConfig(**checkpoint["config"])),
                               attention_mode=checkpoint['settings'].get('attention_mode','causal'))
    model.load_state_dict(checkpoint["classifier"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    x = cache["train"].double(); y = cache["labels"].double(); raw_val = cache["validation"]
    mean = x.mean(0); scale = x.std(0).clamp_min(.05); normalized = (x - mean) / scale
    head = torch.nn.Linear(x.shape[1], 2).double()
    with torch.no_grad():
        head.weight.copy_(model.head.weight.double() * scale)
        head.bias.copy_(model.head.bias.double() + model.head.weight.double() @ mean)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=20, history_size=20, line_search_fn="strong_wolfe")
    args.run_dir.mkdir(parents=True)
    best = None; selected = None; selected_step = None
    for step in range(11):
        if step:
            def closure():
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(normalized), y) + .0001 * head.weight.square().sum()
                if not torch.isfinite(loss): raise ValueError("Loss não finita.")
                loss.backward(); return loss
            optimizer.step(closure)
        fold_normalization(head, model.head, mean, scale)
        result = evaluate_head(model.head, raw_val, validation); key = quality_key(result)
        with torch.no_grad():
            predicted = model.head(cache["train"]) >= 0
            train_exact = (predicted == y.bool()).all(1).float().mean().item()
        print(json.dumps(dict(step=step, train_exact_units=train_exact, correct=result["correct"], quality=key)), flush=True)
        if best is None or key > best:
            best, selected, selected_step = key, result, step
            torch.save(dict(checkpoint, classifier=model.state_dict(), step=step,
                            settings=dict(checkpoint["settings"], optimization="standardized_full_batch_unweighted_lbfgs")), args.run_dir / "best.pt")
    (args.run_dir / "run.json").write_text(json.dumps(dict(info, method="standardized_full_batch_unweighted_lbfgs", parent=str(args.source),
        parent_checkpoint_sha256=hashlib.sha256((args.source / "best.pt").read_bytes()).hexdigest(), lbfgs_iterations_per_outer=20,
        l2_regularization=.0001, threshold=.5, device="cpu", train_normalization_only=True), ensure_ascii=False, indent=2))
    (args.run_dir / "evaluation.json").write_text(json.dumps(dict(selected=selected, selected_step=selected_step, quality=best,
        acceptance=acceptance(selected), adopted=False), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
