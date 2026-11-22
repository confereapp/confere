"""Preserva distribuições corretas do checkpoint próprio durante novos ajustes."""
import torch
from torch.nn import functional as F


def correct_teacher_kl(logits, teacher_logits, labels):
    teacher_logits = teacher_logits.detach()
    mask = (labels != -100) & (teacher_logits.argmax(-1) == labels)
    if not mask.any(): return logits.sum() * 0
    return F.kl_div(F.log_softmax(logits[mask], -1), F.softmax(teacher_logits[mask], -1), reduction="batchmean")
