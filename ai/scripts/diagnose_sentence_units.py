"""Ablação offline de unidades por frase; não altera API nem checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import torch
from confere_ai.model import GPT, GPTConfig
from confere_ai.project_review import review_extracted
from confere_ai.tokenizer import ByteBPE


def split_document(document):
    result = copy.deepcopy(document)
    units = []
    for unit in document["units"]:
        text = unit["text"]
        starts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ🙂])", text)]
        ends = starts[1:] + [len(text)]
        for i, (start, end) in enumerate(zip(starts, ends)):
            units.append(dict(id=f"{unit['id']}:sentence:{i}", text=text[start:end]))
        assert "".join(u["text"] for u in units if u["id"].startswith(unit["id"]+":sentence:")) == text
    result["units"] = units
    assert sum(len(u["text"]) for u in units) == sum(len(u["text"]) for u in document["units"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0: parser.error("Limite deve ser não negativo.")
    if args.output.exists(): parser.error("Saída deve ser nova.")
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config=GPTConfig(**ckpt["config"])
    classifier=None
    if ckpt['settings'].get('objective')=='document_classifier':
        from confere_ai.evidence_classifier import EvidencePredictor
        classifier=EvidencePredictor(ckpt,'mps')
    else:
        model = GPT(config).to("mps")
        model.load_state_dict(ckpt["model"]); model.eval()
    tokenizer = ByteBPE(ckpt["merges"])
    def predict(ids):
        generated = model.generate(torch.tensor([ids], device="mps"), 256, tokenizer.EOS)
        return tokenizer.decode(generated[0, len(ids):].tolist())
    cases = json.loads(args.cases.read_text())
    if args.limit: cases = cases[:args.limit]
    output = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(), case_ids=[c["id"] for c in cases], adopted=False,
                  note="Ablação offline: todas as frases processadas, sem busca por requisito ou descarte de texto. IDs e offsets relativos às novas unidades; não é contrato de produção. Regex não é segmentador geral de português e não trata todas as abreviações. Casos autorais de desenvolvimento.")
    for mode in ["whole", "sentences"]:
        results = []
        correct = false_positive = valid = calls = 0
        for scene in cases:
            document = scene["document"] if mode == "whole" else split_document(scene["document"])
            result = review_extracted([document], scene["requirements"], tokenizer, config.block_size, None if classifier else predict,
                                      schema_version=2, response_tokens=256, predict_units=classifier.predict_units if classifier else None)
            for row, expected in zip(result["requirements"], scene["expected"]):
                passed = row["model_outputs_valid"] and row["status"] == expected["status"] and row["conflicting_evidence"] == expected["conflicting_evidence"]
                correct += passed; valid += row["model_outputs_valid"]
                false_positive += row["status"] == "atendido" and expected["status"] != "atendido"
                calls += row["coverage"]["batches"]
                assert row["coverage"]["processed_characters"] == sum(len(u["text"]) for u in document["units"])
                results.append(dict(case=scene["id"], expected=expected, actual=row["status"], conflict=row["conflicting_evidence"], valid=row["model_outputs_valid"], correct=passed))
            print(json.dumps(dict(mode=mode, case=scene["id"], correct=correct)), flush=True)
        output[mode] = dict(correct=correct, total=len(results), false_positive=false_positive, valid=valid, calls=calls, results=results)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({mode:{k:v for k,v in output[mode].items() if k!='results'} for mode in ['whole','sentences']}), flush=True)


if __name__ == "__main__": main()
