"""Sonda linear sobre representações locais congeladas; não muda a API."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.evidence_classifier import EvidenceClassifier, examples, unit_batch, response
from confere_ai.generative_selection import quality_key
from confere_ai.grounding import judge
from confere_ai.instruct import load_instructions, acceptance
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE


def evaluate_head(head, features, rows):
    with torch.no_grad(): probabilities = head(features).sigmoid().tolist()
    results = []; offset = 0
    for row in rows:
        count = len(row["units"])
        raw = response(row["units"], probabilities[offset:offset+count]); offset += count
        results.append(dict(id=row["id"], task=row["task"], prompt=row["prompt"], expected=row["target"], actual=raw,
                            evaluation_profile=row.get("evaluation_profile", "default"), **judge(raw, row)))
    assert offset == len(features)
    return dict(correct=sum(r["exact_match"] for r in results), total=len(results), samples=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "run-dir"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists(): parser.error("Saída deve ser nova.")
    torch.set_num_threads(4); torch.manual_seed(42)
    rows, digest = load_instructions(args.data)
    if any(not r["task"].startswith("grounded_") for r in rows): parser.error("Somente exemplos documentais.")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(source["merges"])
    backbone = GPT(GPTConfig(**source["config"])); backbone.load_state_dict(source["model"])
    model = EvidenceClassifier(backbone).to("mps").eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    train = examples([r for r in rows if r["split"] == "train"], tokenizer, backbone.config.block_size)
    validation = [r for r in rows if r["split"] == "validation"]
    val = examples(validation, tokenizer, backbone.config.block_size)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(43))[:12000].tolist()
    train = [train[i] for i in indices]
    args.run_dir.mkdir(parents=True)
    def features(items, label):
        chunks = []
        for offset in range(0, len(items), 16):
            ids, lengths = unit_batch(items[offset:offset+16], "mps", backbone.config.block_size)
            with torch.no_grad(): chunks.append(model.features(ids, lengths).cpu())
            if offset % 1024 == 0: print(json.dumps(dict(phase=label, units=offset, total=len(items))), flush=True)
        return torch.cat(chunks)
    x = features(train, "train_features"); y = torch.tensor([r["labels"] for r in train])
    v = features(val, "validation_features")
    torch.save(dict(train=x, labels=y, validation=v, train_sample_indices=indices), args.run_dir / "features.pt")
    head = model.head.cpu()
    for parameter in head.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
    positives = y.sum(0)
    if not torch.all(positives > 0): raise ValueError("As duas classes positivas devem estar presentes.")
    weight = (len(y) - positives) / positives
    generator = torch.Generator().manual_seed(44); best = None; selected = None; selected_step = None
    for step in range(2001):
        if step:
            indices = torch.randint(len(x), (128,), generator=generator)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[indices]), y[indices], pos_weight=weight)
            if not torch.isfinite(loss): raise ValueError("Loss não finita.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if step % 100 == 0:
            result = evaluate_head(head, v, validation); key = quality_key(result)
            print(json.dumps(dict(step=step, correct=result["correct"], total=result["total"], quality=key)), flush=True)
            if best is None or key > best:
                best, selected, selected_step = key, result, step
                torch.save(dict(format_version=1, config=source["config"], merges=source["merges"], classifier=model.state_dict(),
                                step=step, dataset_fingerprint=digest, settings=dict(objective="document_classifier", frozen_backbone=True)),
                           args.run_dir / "best.pt")
    info = dict(method="frozen_backbone_linear_evidence_probe", data=str(args.data), dataset_fingerprint=digest,
                source=str(args.checkpoint), source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                train_units=len(x), validation_units=len(v), test_examples_used=0, threshold=.5,
                note="Referências e JSON são construídos a partir das unidades; somente relevância/estado são previstos. Desenvolvimento, não teste cego. Sem integração na API.")
    (args.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (args.run_dir / "evaluation.json").write_text(json.dumps(dict(selected=selected, selected_step=selected_step, quality=best,
                                                                 acceptance=acceptance(selected), adopted=False), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
