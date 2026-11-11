"""Audita objetivos separados a partir da avaliação pareada já executada."""
import argparse
import hashlib
import json
from pathlib import Path

from confere_ai.instruct import acceptance


def summarize(evaluation):
    before, after = evaluation["baseline"]["samples"], evaluation["selected"]["samples"]
    fields = ("id", "task", "prompt", "expected")
    if not before or [[r[k] for k in fields] for r in before] != [[r[k] for k in fields] for r in after]:
        raise ValueError("A comparação exige a mesma coorte nas duas versões.")
    profiles, conversations = {}, {}
    for previous, current in zip(before, after):
        if current["task"].startswith("grounded_"):
            group = profiles.setdefault(current.get("evaluation_profile", "default"),
                                        dict(total=0, before=0, after=0, valid=0, status=0, pointers=0, false_positive=0))
            group["total"] += 1; group["before"] += previous["exact_match"]; group["after"] += current["exact_match"]
            group["valid"] += current["valid_format"] and current["reference_valid"]
            group["status"] += current["status_correct"]; group["pointers"] += current["unit_correct"]
            group["false_positive"] += (current["result"]["status"] == "atendido"
                                        and json.loads(current["expected"])["status"] != "atendido")
        elif current["task"].startswith("conversation_"):
            group = conversations.setdefault(current["task"], dict(total=0, before=0, after=0))
            group["total"] += 1; group["before"] += previous["exact_match"]; group["after"] += current["exact_match"]
    regression_pass = bool(conversations) and all(g["after"] >= g["before"] and g["after"]/g["total"] >= .8
                                                 for g in conversations.values())
    criteria = acceptance(evaluation["selected"])
    return dict(profiles=profiles, conversations=conversations, acceptance=criteria,
                conversation_regression_pass=regression_pass, controlled_gate_pass=bool(profiles) and criteria["passed"] and regression_pass,
                adopted=False, test_examples_used=0,
                note="Validação sintética de desenvolvimento. Nenhuma promoção automática, mesmo se o critério controlado passar.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Saída já existe.")
    info = json.loads((args.run_dir / "run.json").read_text())
    evaluation = json.loads((args.run_dir / "instruction-evaluation.json").read_text())
    if evaluation["test_examples_used"] or info["test_examples_used"]:
        parser.error("Este relatório é restrito à validação.")
    if hashlib.sha256(Path(info["data"]).read_bytes()).hexdigest() != info["dataset_fingerprint"]:
        parser.error("O corpus mudou desde a execução.")
    result = summarize(evaluation)
    result.update(dataset_sha256=info["dataset_fingerprint"], selected_step=evaluation["selected_step"],
                  checkpoint_sha256=hashlib.sha256((args.run_dir / "best.pt").read_bytes()).hexdigest(),
                  context_size=info["paired_evaluation_context"], response_tokens=info["paired_evaluation_response_tokens"])
    with args.output.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
