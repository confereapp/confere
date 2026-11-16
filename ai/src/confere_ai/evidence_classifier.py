"""Cabeça de evidência por unidade; decisões semânticas continuam aprendidas."""
import json
import hashlib
import re
import torch
from torch import nn
from .grounding import prompt_for
from .tokenizer import ByteBPE
from .model import GPT, GPTConfig

STATE_MODES = ("independent_binary", "categorical_four_state")
READOUTS = ("shared_json_decision_prefix_v1", "suffix_query_decision_prefix_v1")


def sentence_parts(text):
    # shortcut: regex preserva o texto, mas abreviações exigem segmentação linguística antes de uso geral.
    starts=[0]+[m.end() for m in re.finditer(r"[.!?]\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ🙂])",text)]
    result=[text[a:b] for a,b in zip(starts,starts[1:]+[len(text)])]
    assert ''.join(result)==text
    return result


class EvidenceClassifier(nn.Module):
    def __init__(self, backbone, state_mode=STATE_MODES[0]):
        super().__init__()
        if state_mode not in STATE_MODES: raise ValueError("Modo de estados não suportado.")
        self.state_mode = state_mode
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.width, 4 if state_mode == STATE_MODES[1] else 2)
        backbone.initialize(self.head)

    def features(self, ids, lengths):
        hidden = self.backbone(ids)
        return hidden[torch.arange(len(ids), device=ids.device), lengths - 1]

    def forward(self, ids, lengths):
        return self.head(self.features(ids, lengths))

    def evidence_scores(self, logits):
        if self.state_mode == STATE_MODES[0]: return logits.sigmoid()
        states = logits.argmax(-1)
        return torch.stack(((states & 1) != 0, (states & 2) != 0), dim=-1).float()

    def loss(self, logits, labels):
        if self.state_mode == STATE_MODES[0]:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        states = (labels[:, 0] + 2 * labels[:, 1]).long()
        return torch.nn.functional.cross_entropy(logits, states)


@torch.no_grad()
def load_binary_as_categorical(model, state):
    if model.state_mode != STATE_MODES[1]: raise ValueError("Destino deve usar quatro estados.")
    missing, unexpected = model.load_state_dict({k:v for k,v in state.items() if not k.startswith("head.")}, strict=False)
    assert set(missing) == {"head.weight", "head.bias"} and not unexpected
    for name in ["weight", "bias"]:
        source = state["head."+name]; target = getattr(model.head, name)
        target[0].zero_(); target[1].copy_(source[0]); target[2].copy_(source[1]); target[3].copy_(source[0]+source[1])


def decision_prefix(tokenizer):
    choices = [tokenizer.encode(json.dumps(dict(status=status, evidence=[]), ensure_ascii=False))
               for status in ("atendido", "nao_atendido", "evidencia_insuficiente")]
    prefix = []
    for values in zip(*choices):
        if len(set(values)) != 1: break
        prefix.append(values[0])
    return prefix


def encode_unit(requirement, text, tokenizer, context_size, prefix=None, *, readout=READOUTS[0]):
    if readout not in READOUTS: raise ValueError("Representação do classificador não suportada.")
    prompt = (prompt_for(requirement, [dict(id="page:1", text=text)], 2) if readout == READOUTS[0]
              else f"Texto a conferir:\n{text}\nRequisito consultado: {requirement}")
    prefix = decision_prefix(tokenizer) if prefix is None else prefix
    ids = [tokenizer.BOS] + tokenizer.encode(prompt) + [tokenizer.SEP] + prefix
    if len(ids) > context_size:
        raise ValueError("Unidade excede o contexto; nenhum texto foi truncado.")
    return ids


def examples(rows, tokenizer, context_size, *, readout=READOUTS[0]):
    result = []
    prefix = decision_prefix(tokenizer)
    for row in rows:
        target = json.loads(row["target"])
        if row.get("grounding_schema") != 2: raise ValueError("Classificador exige contrato V2.")
        pointers = {(e["unit"], e["status"]) for e in target["evidence"]}
        for unit in row["units"]:
            labels = [float((unit["id"], status) in pointers) for status in ("atendido", "nao_atendido")]
            result.append(dict(row_id=row["id"], unit=unit["id"], labels=labels, source_key=hashlib.sha256(unit["text"].encode()).hexdigest(),
                               ids=encode_unit(row["requirement"], unit["text"], tokenizer, context_size, prefix, readout=readout)))
    return result



def paired_indices(items, pools, batch_size, generator, *, label_key="labels"):
    if batch_size <= 0 or batch_size % 2: raise ValueError("Lote pareado exige tamanho positivo e par.")
    if not pools: raise ValueError("Fontes pareadas vazias.")
    result = []
    for group in torch.randint(len(pools), (batch_size // 2,), generator=generator).tolist():
        pool = pools[group]
        first = pool[torch.randint(len(pool), (), generator=generator).item()]
        alternatives = [i for i in pool if items[i][label_key] != items[first][label_key]]
        choices = alternatives or pool
        second = choices[torch.randint(len(choices), (), generator=generator).item()]
        result.extend([first, second])
    return result

def unit_batch(items, device, context_size):
    lengths = [len(item["ids"]) for item in items]
    length = max(lengths)
    if torch.device(device).type == "mps": length = min(context_size, ((length + 127) // 128) * 128)
    ids = torch.full((len(items), length), ByteBPE.PAD, dtype=torch.long)
    for i, item in enumerate(items): ids[i, :lengths[i]] = torch.tensor(item["ids"])
    return ids.to(device), torch.tensor(lengths, device=device)


def response(units, probabilities, thresholds=(.5, .5)):
    if len(thresholds) != 2 or any(not isinstance(v, (float, int)) or not 0 < v < 1 for v in thresholds):
        raise ValueError("Limiares devem conter dois números entre zero e um.")
    if len(probabilities) != len(units): raise ValueError("Uma previsão é obrigatória por unidade.")
    evidence = []
    for unit, values in zip(units, probabilities):
        if len(values) != 2 or any(not 0 <= value <= 1 for value in values):
            raise ValueError("Probabilidades inválidas.")
        for status, value, threshold in zip(("atendido", "nao_atendido"), values, thresholds):
            if value >= threshold: evidence.append(dict(status=status, unit=unit["id"]))
    statuses = {e["status"] for e in evidence}
    status = next(iter(statuses)) if len(statuses) == 1 else "evidencia_insuficiente"
    return json.dumps(dict(status=status, evidence=evidence), ensure_ascii=False)


@torch.no_grad()
def fold_normalization(source, target, mean, scale):
    if not torch.all(scale > 0): raise ValueError("Escalas devem ser positivas.")
    weights = source.weight / scale
    target.weight.copy_(weights.to(target.weight.dtype))
    target.bias.copy_((source.bias - weights @ mean).to(target.bias.dtype))


class EvidencePredictor:
    def __init__(self, checkpoint, device):
        settings = checkpoint["settings"]
        if (settings.get("objective") != "document_classifier" or settings.get("readout") not in READOUTS
            or settings.get("threshold", .5) != .5):
            raise ValueError("Contrato do classificador não suportado.")
        self.thresholds = settings.get("thresholds", [.5, .5])
        response([], [], self.thresholds)
        self.readout = settings["readout"]
        self.segmentation=settings.get('unit_segmentation','whole_unit_v1')
        if self.segmentation not in {'whole_unit_v1','sentences_v1'}: raise ValueError('Segmentação de unidade não suportada.')
        self.tokenizer = ByteBPE(checkpoint["merges"])
        self.config = GPTConfig(**checkpoint["config"])
        if self.tokenizer.vocab_size != self.config.vocab_size: raise ValueError("Tokenizer incompatível.")
        self.model = EvidenceClassifier(GPT(self.config), settings.get("state_mode", STATE_MODES[0]))
        self.model.load_state_dict(checkpoint["classifier"])
        self.device = device; self.model.to(device).eval()

    @torch.no_grad()
    def probabilities(self, requirement, units):
        if not units or len(units) > 4 or len({u["id"] for u in units}) != len(units):
            raise ValueError("Informe de uma a quatro unidades distintas.")
        parts=[sentence_parts(u['text']) if self.segmentation=='sentences_v1' else [u['text']] for u in units]
        items = [dict(ids=encode_unit(requirement, text, self.tokenizer, self.config.block_size, readout=self.readout)) for group in parts for text in group]
        predictions=[]
        for offset in range(0,len(items),8):
            ids,lengths=unit_batch(items[offset:offset+8],self.device,self.config.block_size)
            predictions.extend(self.model.evidence_scores(self.model(ids,lengths)).cpu().tolist())
        if any(len(v)!=2 or any(not 0<=s<=1 for s in v) for v in predictions):
            raise ValueError('Probabilidades de frase inválidas.')
        probabilities=[];offset=0
        for group in parts:
            values=predictions[offset:offset+len(group)];offset+=len(group)
            probabilities.append([max(v[i] for v in values) for i in range(2)])
        return probabilities

    def predict_units(self, requirement, units):
        return response(units,self.probabilities(requirement,units),self.thresholds)
