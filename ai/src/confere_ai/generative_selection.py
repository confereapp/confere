"""Ordena candidatos por respostas geradas; aceitação continua por família."""
import json
from collections import defaultdict


def quality_key(result):
    groups = defaultdict(lambda: [0, 0])
    invalid = false_positive = 0
    for row in result["samples"]:
        documentary = row["task"].startswith("grounded_")
        if not documentary and not row["task"].startswith("conversation_"):
            raise ValueError("Seleção gerativa exige tarefas controladas de conversa/documentos.")
        threshold = .9 if documentary or row["task"].startswith("conversation_context_answer_") else .8
        name = row.get("evaluation_profile", "default") if documentary else row["task"]
        pair = groups[("documents" if documentary else "conversation", name, threshold)]
        pair[0] += row["exact_match"]; pair[1] += 1
        if documentary:
            invalid += not (row["valid_format"] and row["reference_valid"])
            false_positive += row["result"]["status"] == "atendido" and json.loads(row["expected"])["status"] != "atendido"
    if not groups: raise ValueError("Coorte gerativa vazia.")
    ratios = [correct / total / threshold for (_, _, threshold), (correct, total) in groups.items()]
    worst = min(ratios)
    passed = not invalid and not false_positive and worst >= 1
    return (passed, -invalid, worst, sum(min(1., ratio) for ratio in ratios) / len(ratios),
            -false_positive, sum(row["exact_match"] for row in result["samples"]))
    ratios = [min(1., correct / total / threshold) for (_, _, threshold), (correct, total) in groups.items()]
    return (int(not invalid and not false_positive and min(ratios) >= 1), -invalid,
            min(ratios), sum(ratios) / len(ratios), -false_positive, result["correct"])
