"""Supervisão de tipo de resposta, usada apenas durante o treino."""
import torch
from torch.nn import functional as F


def semantic_class(row):
    if not row['task'].startswith('conversation_context_answer_') or 'memory' in row['task']:
        return 5
    return {'Não tenho informação suficiente para responder.':1,'Há informações conflitantes no texto.':2,
            'Sim.':3,'Não.':4}.get(row['target'],0)


def semantic_forward(model,ids,labels,head,classes):
    captured=[]
    handle=model.norm.register_forward_hook(lambda module,inputs,output:captured.append(output))
    try:logits=model(ids)
    finally:handle.remove()
    if (labels!=-100).sum(-1).min()==0:raise ValueError('Resposta sem tokens supervisionados.')
    positions=(labels!=-100).long().argmax(-1)
    features=captured[0][torch.arange(len(ids),device=ids.device),positions]
    return logits,F.cross_entropy(head(features),classes)
