"""Cabeça de evidência por unidade; decisões semânticas continuam aprendidas."""
import json
import torch
from torch import nn
from .grounding import prompt_for
from .tokenizer import ByteBPE


class EvidenceClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.width, 2)
        backbone.initialize(self.head)

    def features(self, ids, lengths):
        hidden = self.backbone(ids)
        return hidden[torch.arange(len(ids), device=ids.device), lengths - 1]

    def forward(self, ids, lengths):
        return self.head(self.features(ids, lengths))


def decision_prefix(tokenizer):
    choices = [tokenizer.encode(json.dumps(dict(status=status, evidence=[]), ensure_ascii=False))
               for status in ("atendido", "nao_atendido", "evidencia_insuficiente")]
    prefix = []
    for values in zip(*choices):
        if len(set(values)) != 1: break
        prefix.append(values[0])
    return prefix


def encode_unit(requirement, text, tokenizer, context_size, prefix=None):
    prompt = prompt_for(requirement, [dict(id="page:1", text=text)], 2)
    prefix = decision_prefix(tokenizer) if prefix is None else prefix
    ids = [tokenizer.BOS] + tokenizer.encode(prompt) + [tokenizer.SEP] + prefix
    if len(ids) > context_size:
        raise ValueError("Unidade excede o contexto; nenhum texto foi truncado.")
    return ids


def examples(rows, tokenizer, context_size):
    result = []
    prefix = decision_prefix(tokenizer)
    for row in rows:
        target = json.loads(row["target"])
        if row.get("grounding_schema") != 2: raise ValueError("Classificador exige contrato V2.")
        pointers = {(e["unit"], e["status"]) for e in target["evidence"]}
        for unit in row["units"]:
            labels = [float((unit["id"], status) in pointers) for status in ("atendido", "nao_atendido")]
            result.append(dict(row_id=row["id"], unit=unit["id"], labels=labels,
                               ids=encode_unit(row["requirement"], unit["text"], tokenizer, context_size, prefix)))
    return result


def unit_batch(items, device, context_size):
    lengths = [len(item["ids"]) for item in items]
    length = max(lengths)
    if torch.device(device).type == "mps": length = min(context_size, ((length + 127) // 128) * 128)
    ids = torch.full((len(items), length), ByteBPE.PAD, dtype=torch.long)
    for i, item in enumerate(items): ids[i, :lengths[i]] = torch.tensor(item["ids"])
    return ids.to(device), torch.tensor(lengths, device=device)


def response(units, probabilities):
    if len(probabilities) != len(units): raise ValueError("Uma previsão é obrigatória por unidade.")
    evidence = []
    for unit, values in zip(units, probabilities):
        if len(values) != 2 or any(not 0 <= value <= 1 for value in values):
            raise ValueError("Probabilidades inválidas.")
        for status, value in zip(("atendido", "nao_atendido"), values):
            if value >= .5: evidence.append(dict(status=status, unit=unit["id"]))
    statuses = {e["status"] for e in evidence}
    status = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
    return json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False)


@torch.no_grad()
def fold_normalization(source, target, mean, scale):
    if not torch.all(scale > 0): raise ValueError("Escalas devem ser positivas.")
    weights = source.weight / scale
    target.weight.copy_(weights.to(target.weight.dtype))
    target.bias.copy_((source.bias - weights @ mean).to(target.bias.dtype))
