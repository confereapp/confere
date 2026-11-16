"""Avalia documentos anotados completos pelo mesmo pipeline da API textual."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from confere_ai.grounding import judge
from confere_ai.model import GPT, GPTConfig
from confere_ai.project_review import review_extracted
from confere_ai.tokenizer import ByteBPE
from confere_ai.train import choose_device
from prepare_api_batches import local_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Saída já existe.")
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        parser.error("Informe casos anotados não vazios.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    is_classifier = checkpoint["settings"].get("objective") == "document_classifier"
    if not is_classifier and (checkpoint["settings"].get("grounding_schema") != 2 or checkpoint["settings"].get("generation_budget") != 256):
        parser.error("Avaliação exige contrato V2 e orçamento de 256 tokens.")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    config = GPTConfig(**checkpoint["config"])
    classifier = None
    if is_classifier:
        from confere_ai.evidence_classifier import EvidencePredictor
        classifier = EvidencePredictor(checkpoint, device)
    else:
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model"]); model.eval()
    tokenizer = ByteBPE(checkpoint["merges"])
    def predict(ids):
        output = model.generate(torch.tensor([ids], device=device), 256, tokenizer.EOS)
        return tokenizer.decode(output[0, len(ids):].tolist())
    results = []
    total = correct = local_total = local_correct = false_positive = valid = 0
    for scene in cases:
        document = scene["document"]; text = document["units"][0]["text"]
        assert hashlib.sha256(text.encode()).hexdigest() == document["sha256"]
        assert all(text[a["start"]:a["end"]] == a["quote"] for a in scene["annotations"])
        expected_rows = local_rows(scene, tokenizer, config.block_size)
        if expected_rows is None:
            raise ValueError("Evidência parcial: o tokenizer não corresponde ao corpus validado.")
        result = review_extracted([document], scene["requirements"], tokenizer, config.block_size,
                                  None if classifier else predict, schema_version=2, response_tokens=256,
                                  predict_units=classifier.predict_units if classifier else None)
        assert len(scene["requirements"]) == len(scene["expected"]) == len(result["requirements"])
        requirement_results = []; cursor = 0
        for row, expected in zip(result["requirements"], scene["expected"]):
            sources = [s for call in row["calls"] for s in call["sources"]]
            assert sources[0]["char_start"] == 0 and sources[-1]["char_end"] == len(text)
            assert all(a["char_end"] == b["char_start"] for a, b in zip(sources, sources[1:]))
            for evidence in row["evidence"]:
                assert evidence["quote"] == text[evidence["char_start"]:evidence["char_end"]]
            judgments = [judge(call["raw_model_output"], expected_rows[cursor+i]) for i, call in enumerate(row["calls"])]
            cursor += len(row["calls"])
            sound = row["model_outputs_valid"]
            passed = sound and row["status"] == expected["status"] and row["conflicting_evidence"] == expected["conflicting_evidence"]
            total += 1; correct += passed; valid += sound
            false_positive += row["status"] == "atendido" and expected["status"] != "atendido"
            local_total += len(judgments); local_correct += sum(j["exact_match"] for j in judgments)
            requirement_results.append(dict(expected=expected, actual=row["status"], conflict=row["conflicting_evidence"],
                                            correct=passed, valid=sound, local_correct=sum(j["exact_match"] for j in judgments),
                                            local_total=len(judgments)))
        assert cursor == len(expected_rows)
        results.append(dict(id=scene["id"], profile=scene["profile"], requirements=requirement_results, trace=result))
        print(json.dumps(dict(id=scene["id"], correct=sum(r["correct"] for r in requirement_results), total=len(requirement_results))), flush=True)
    output = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  prediction_type="unit_classifier" if is_classifier else "autoregressive",
                  step=checkpoint["step"], cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  correct_requirements=correct, total_requirements=total, valid_requirements=valid,
                  false_positive=false_positive, exact_local_calls=local_correct, total_local_calls=local_total,
                  complete_projects=sum(all(r["correct"] for r in case["requirements"]) for case in results),
                  total_projects=len(results), test_examples_used=0, adopted=False, results=results,
                  note="Documentos autorais da validação de desenvolvimento; sobrepõem os lotes locais da avaliação. Sem teste cego ou documento real.")
    with args.output.open("x") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
