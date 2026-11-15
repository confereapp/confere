"""Gates separados para respostas contextuais e regressões; não valida conversa geral."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from confere_ai.instruct import acceptance


def groups(samples):
    counts = defaultdict(lambda: [0, 0])
    for row in samples:
        counts[row["task"]][0] += row["exact_match"]
        counts[row["task"]][1] += 1
    return dict(counts)


def audit(evaluation):
    before, after = evaluation["baseline"]["samples"], evaluation["selected"]["samples"]
    assert [(r["id"], r["prompt"], r["expected"]) for r in before] == [(r["id"], r["prompt"], r["expected"]) for r in after]
    assert evaluation["test_examples_used"] == 0
    before_groups, after_groups = groups(before), groups(after)
    context = {k: v for k, v in after_groups.items() if k.startswith("conversation_context_answer_")}
    assert len(context) == 5 and all(v[1] for v in context.values())
    context_pass = all(correct / total >= .9 for correct, total in context.values())
    regressions = {task: dict(before=before_groups[task], after=values) for task, values in after_groups.items()
                   if not task.startswith("conversation_context_answer_") and values[0] < before_groups[task][0]}
    docs = [r for r in after if r["task"].startswith("grounded_")]
    assert docs
    doc_gate = acceptance({"samples": docs})
    return dict(context_families=context, context_pass=context_pass, regressions=regressions,
                document_gate=doc_gate, controlled_criteria_pass=context_pass and not regressions and doc_gate["passed"],
                adopted=False, note="Gates de desenvolvimento: >=90% em cada família contextual, sem regressão por família legada e critério documental pré-existente. Mesmo aprovados, não comprovam conversa geral ou documentos reais.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()
    path = args.run_dir / ("context-audit.json" if args.checkpoint == "best" else "context-audit-last.json")
    if path.exists(): parser.error("Auditoria já existe.")
    evaluation = json.loads((args.run_dir / "instruction-evaluation.json").read_text())
    if args.checkpoint == "last":
        import torch
        from confere_ai.instruct import load_instructions, score
        from confere_ai.model import GPT, GPTConfig
        from confere_ai.tokenizer import ByteBPE
        info = json.loads((args.run_dir / "run.json").read_text())
        rows, digest = load_instructions(Path(info["data"]))
        assert digest == info["dataset_fingerprint"]
        checkpoint = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=True)
        model = GPT(GPTConfig(**checkpoint["config"])).to("mps")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        torch.set_num_threads(4)
        samples = [r for r in rows if r["split"] == "validation"]
        evaluation = dict(baseline=evaluation["baseline"], selected=score(model, ByteBPE(checkpoint["merges"]), samples,
                          "mps", context_limit=1024, response_tokens=256), selected_step=checkpoint["step"], test_examples_used=0)
        with (args.run_dir / "last-evaluation.json").open("x") as file:
            json.dump(evaluation, file, ensure_ascii=False, indent=2)
    result = audit(evaluation)
    result.update(selected_step=evaluation["selected_step"], checkpoint_sha256=hashlib.sha256((args.run_dir / (args.checkpoint + ".pt")).read_bytes()).hexdigest())
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
