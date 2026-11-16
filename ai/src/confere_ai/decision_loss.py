"""Peso extra nos valores documentais; o protocolo de avaliação não muda."""
import re

import torch
from torch.nn import functional as F


def decision_weights(row, tokenizer, labels, factor):
    weights = [0. if label == -100 else 1. for label in labels]
    if not row["task"].startswith("grounded_"):
        return weights
    target = row["target"].encode("utf-8")
    spans = [match.span(1) for match in re.finditer(rb'"(?:status|unit)"\s*:\s*"([^"\\]*)"', target)]
    if not spans:
        raise ValueError("Alvo documental sem valores de decisão.")
    tokens = tokenizer.encode(row["target"])
    offset = len(labels) - len(tokens) - 1
    if labels[offset:] != tokens + [tokenizer.EOS]:
        raise ValueError("Alvo e labels desalinhados.")
    position = 0
    for index, token in enumerate(tokens):
        end = position + len(tokenizer.vocab[token])
        if any(position < right and end > left for left, right in spans):
            weights[offset + index] = factor
        position = end
    assert position == len(target)
    return weights


def weighted_loss(logits, labels, weights):
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none")
    active = weights.flatten() * (labels.flatten() != -100)
    return (losses * active).sum() / active.sum()


def weight_batch(examples, indices, shape, device):
    weights = torch.zeros(shape)
    for index, source in enumerate(indices):
        values = examples[source]
        weights[index, :len(values)] = torch.tensor(values)
    return weights.to(device)
