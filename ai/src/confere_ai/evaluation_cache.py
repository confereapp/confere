"""Reaproveitamento explícito de geração com pesos e coorte exatamente iguais."""
import hashlib
import json
from pathlib import Path


def protocol_fingerprint():
    import inspect
    import torch
    from .instruct import score
    root = Path(__file__).parent
    payload = b"".join((root / name).read_bytes() for name in ["model.py", "chat.py", "tokenizer.py"])
    return hashlib.sha256(payload + inspect.getsource(score).encode() + str(torch.__version__).encode()).hexdigest()


def reuse_baseline(directory, checkpoint, validation, context_size, response_tokens, device="mps"):
    from .instruct import load_instructions
    directory = Path(directory)
    info = json.loads((directory / "run.json").read_text())
    audit = json.loads((directory / "context-audit.json").read_text())
    result = json.loads((directory / "instruction-evaluation.json").read_text())
    protocol = info.get("evaluation_protocol_hash")
    sidecar = directory / "evaluation-protocol.json"
    if protocol is None and sidecar.exists():
        protocol = json.loads(sidecar.read_text())["evaluation_protocol_hash"]
    if protocol != protocol_fingerprint():
        raise ValueError("Código ou runtime de geração do cache não foi verificado ou mudou.")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != audit["checkpoint_sha256"] or result["selected_step"] != audit["selected_step"]:
        raise ValueError("Cache não corresponde aos pesos selecionados.")
    if (info["device"] != device or info["paired_evaluation_context"] != context_size
        or info["paired_evaluation_response_tokens"] != response_tokens or result["test_examples_used"] != 0):
        raise ValueError("Cache usa protocolo de geração diferente.")
    rows, source_digest = load_instructions(Path(info["data"]))
    if source_digest != info["dataset_fingerprint"]:
        raise ValueError("Corpus original do cache foi alterado.")
    old = [r for r in rows if r["split"] == "validation" and (response_tokens <= 64 or r["task"] not in {"human_dialogue", "public_dialogue"})]
    if old != validation:
        raise ValueError("Coorte do cache difere, inclusive histórico ou metadados.")
    samples = result["selected"]["samples"]
    if ([(r["id"], r["task"], r["prompt"], r["target"]) for r in validation]
        != [(r["id"], r["task"], r["prompt"], r["expected"]) for r in samples]
        or result["selected"]["total"] != len(samples)
        or result["selected"]["correct"] != sum(r["exact_match"] for r in samples)):
        raise ValueError("Resultados do cache não correspondem à coorte declarada.")
    return result["selected"]
