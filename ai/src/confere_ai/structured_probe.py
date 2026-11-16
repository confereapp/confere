"""Decodificação experimental por probabilidade de respostas JSON completas; não usada pela API."""
import itertools
import json
import torch
from torch.nn import functional as F


def candidates(aliases):
    if not 1 <= len(aliases) <= 4 or aliases != [f"page:{i+1}" for i in range(len(aliases))]:
        raise ValueError("Informe de um a quatro aliases internos consecutivos.")
    for states in itertools.product(range(4), repeat=len(aliases)):
        evidence = []
        for alias, state in zip(aliases, states):
            if state & 1: evidence.append(dict(status="atendido", unit=alias))
            if state & 2: evidence.append(dict(status="nao_atendido", unit=alias))
        statuses = {r["status"] for r in evidence}
        status = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
        yield json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False)


@torch.no_grad()
def choose(model, tokenizer, prefix, aliases, device="mps", response_tokens=256, batch_size=8):
    if batch_size < 1: raise ValueError("Batch deve ser positivo.")
    if len(prefix) < 2 or prefix[0] != tokenizer.BOS or prefix[-1] != tokenizer.SEP:
        raise ValueError("Prefixo deve seguir o protocolo BOS/pergunta/SEP.")
    options = []
    for text in candidates(aliases):
        target = tokenizer.encode(text) + [tokenizer.EOS]
        if len(target) <= response_tokens and len(prefix)+len(target)-1 <= model.config.block_size:
            options.append((text, target))
    if not options: raise ValueError("Nenhuma resposta completa cabe no contexto.")
    best = None
    for offset in range(0, len(options), batch_size):
        chunk = options[offset:offset+batch_size]
        length = max(len(prefix)+len(target)-1 for _, target in chunk)
        if torch.device(device).type == "mps": length = min(model.config.block_size, ((length+127)//128)*128)
        x = torch.full((len(chunk), length), tokenizer.PAD, dtype=torch.long, device=device)
        y = torch.full_like(x, -100)
        for i, (_, target) in enumerate(chunk):
            sequence = prefix + target
            x[i, :len(sequence)-1] = torch.tensor(sequence[:-1], device=device)
            y[i, len(prefix)-1:len(sequence)-1] = torch.tensor(target, device=device)
        losses = F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=-100, reduction="none").sum(dim=1)
        if not torch.isfinite(losses).all().item(): raise ValueError("Probabilidade não finita.")
        for (text, _), loss in zip(chunk, losses.tolist()):
            if best is None or loss < best[0]: best = (loss, text)
    return best[1]
