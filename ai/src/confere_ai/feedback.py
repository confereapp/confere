"""Justificativa por requisito a partir da saída do classificador, sem gerar texto com modelo.

O classificador só decide o estado e aponta trechos; não explica. O texto abaixo descreve o que ele fez
(quantos trechos sustentam, em que páginas) e nunca afirma que um requisito falta. Testamos gerar a frase com
o modelo de conversa próprio e as respostas saíram incoerentes ou genéricas (ai/research/ARTIGO_TRECHOS_IA.md).
"""
import re

TOP_K = 8


def place(unit):
    match = re.fullmatch(r"(page|paragraph):(\d+)", unit)
    if not match: return unit
    return f"p. {match[2]}" if match[1] == "page" else f"parágrafo {match[2]}"


def places(units):
    seen = list(dict.fromkeys(place(u) for u in units))
    return seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " e " + seen[-1]


def explain(status, evidence):
    """status: atendido | nao_atendido | evidencia_insuficiente. evidence: [{source_unit, quote, model_status}]."""
    support = [e for e in evidence if e["model_status"] == status]
    if status == "atendido" and support:
        n = len(support)
        return (f"Proposta automática: atende. {n} {'trecho sustenta' if n == 1 else 'trechos sustentam'} o requisito "
                f"({places(e['source_unit'] for e in support)}). Confira a citação antes de aceitar.")
    if status == "nao_atendido" and support:
        return (f"O classificador leu um trecho que indicaria o contrário ({places(e['source_unit'] for e in support)}). "
                "Não é reprovação automática: revise manualmente.")
    return (f"O classificador não encontrou evidência suficiente entre os {TOP_K} trechos mais relevantes do documento. "
            "Isso não indica que o requisito falta: revise manualmente.")
