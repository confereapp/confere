"""Ponteiros verificáveis: citações vêm da fonte, nunca são reescritas pela rede."""
import json

STATUSES = {"atendido", "nao_atendido", "evidencia_insuficiente"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Chaves duplicadas na resposta.")
        result[key] = value
    return result


def prompt_for(requirement, units, schema_version=1):
    instruction = ("Confira o requisito. Retorne JSON com status e unit." if schema_version == 1 else
                   "Confira o requisito. Retorne JSON com status e evidence (lista de status e unit). Liste todas as evidências relevantes, inclusive opostas. Sem evidência ou com conflito, status evidencia_insuficiente.")
    return instruction + "\nRequisito: " + requirement + "\n" + "\n".join(f'[{unit["id"]}] {unit["text"]}' for unit in units)


def resolve(raw, document, units):
    fallback = {"status": "evidencia_insuficiente", "evidence": [], "justification": "Saída inválida do modelo; conferência inconclusiva."}
    result = {"valid_format": False, "reference_valid": False, "result": fallback}
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return result
    if isinstance(value, dict) and "evidence" in value:
        return resolve_evidence(value, document, units, result)
    if not isinstance(value, dict) or set(value) != {"status", "unit"} or not isinstance(value["status"], str) or value["status"] not in STATUSES:
        return result
    status, unit_id = value["status"], value["unit"]
    if (status == "evidencia_insuficiente" and unit_id is not None) or (status != "evidencia_insuficiente" and not isinstance(unit_id, str)):
        return result
    result["valid_format"] = True
    if unit_id is None:
        result["reference_valid"] = True
        result["result"] = {"status": status, "evidence": [], "justification": "Modelo não encontrou evidência suficiente no contexto fornecido."}
        return result
    matches = [unit for unit in units if unit["id"] == unit_id and unit["text"].strip()]
    if len(matches) != 1:
        return result
    result["reference_valid"] = True
    result["result"] = {"status": status, "evidence": [{"document": document, "unit": unit_id, "quote": matches[0]["text"]}],
                        "justification": "Classificação experimental do modelo; trecho selecionado na fonte. Suporte semântico ainda exige verificação."}
    return result


def resolve_evidence(value, document, units, result):
    if set(value) != {"status", "evidence"} or not isinstance(value["evidence"], list) or len(value["evidence"]) > 8:
        return result
    evidence = value["evidence"]
    seen = set()
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {"status", "unit"}
            or not isinstance(item["status"], str) or item["status"] not in {"atendido", "nao_atendido"}
            or not isinstance(item["unit"], str) or (item["status"], item["unit"]) in seen):
            return result
        seen.add((item["status"], item["unit"]))
    statuses = {item["status"] for item in evidence}
    consolidated = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
    if value["status"] != consolidated:
        return result
    result["valid_format"] = True
    quotes = []
    for item in evidence:
        matches = [u for u in units if u["id"] == item["unit"] and u["text"].strip()]
        if len(matches) != 1:
            return result
        quotes.append(dict(document=document, unit=item["unit"], quote=matches[0]["text"], model_status=item["status"]))
    result["reference_valid"] = True
    result["result"] = dict(status=consolidated, evidence=quotes,
                            justification="Classificação experimental; suporte semântico exige revisão humana.")
    return result


def judge(raw, row):
    resolved = resolve(raw, row["document"], row["units"])
    expected = resolve(row["target"], row["document"], row["units"])
    valid = resolved["valid_format"] and resolved["reference_valid"] and expected["valid_format"] and expected["reference_valid"]
    status_correct = valid and resolved["result"]["status"] == expected["result"]["status"]
    def pointers(result):
        return {(e.get("model_status", result["status"]), e["unit"]) for e in result["evidence"]}
    unit_correct = valid and pointers(resolved["result"]) == pointers(expected["result"])
    return {**resolved, "status_correct": bool(status_correct), "unit_correct": bool(unit_correct),
            "exact_match": bool(status_correct and unit_correct)}
