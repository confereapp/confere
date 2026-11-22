"""Conversa local via JSON por stdin, sem persistência de histórico."""
import argparse
import json
import sys
from pathlib import Path

from .chat import prepare_prompt
from .tokenizer import ByteBPE


def parse_request(raw):
    if len(raw) > 200000:
        raise ValueError("Entrada excede 200 KB.")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) - {"message", "history"}:
        raise ValueError("Informe message e history opcional.")
    def valid_text(text):
        return isinstance(text, str) and bool(text.strip()) and len(text) <= 2000
    if not valid_text(value.get("message")):
        raise ValueError("Mensagem deve ter de 1 a 2000 caracteres.")
    history = value.get("history", [])
    if (not isinstance(history, list) or len(history) > 8
        or any(not isinstance(turn, dict) or set(turn) != {"message", "response"}
               or not valid_text(turn["message"])
               or not isinstance(turn["response"], str) or len(turn["response"]) > 2000 for turn in history)):
        raise ValueError("Histórico deve conter até oito pares de mensagem e resposta.")
    return dict(message=value["message"], history=history)


def converse(request, tokenizer, respond, context_size, response_tokens, objective):
    history = [(turn["message"], turn["response"]) for turn in request["history"]]
    try:
        ids, dropped = prepare_prompt(history, request["message"], tokenizer,
                                      context_size, response_tokens, objective)
    except ValueError:
        return {"error": "context_length"}
    return dict(response=respond(ids), experimental=True, dropped_turns=dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    try:
        request = parse_request(sys.stdin.buffer.read(200001))
        import torch
        from .model import GPT, GPTConfig
        from .train import choose_device

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        tokenizer = ByteBPE(checkpoint["merges"])
        device = choose_device(args.device)
        model = GPT(GPTConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        response_tokens = min(128, model.config.block_size - 3)
        if response_tokens < 1:
            raise ValueError("Contexto inválido.")
        def respond(ids):
            generated = model.generate(torch.tensor([ids], device=device), response_tokens, tokenizer.EOS)
            return tokenizer.decode(generated[0, len(ids):].tolist())
        result = converse(request, tokenizer, respond, model.config.block_size,
                          response_tokens, checkpoint.get("settings", {}).get("objective", "supervised"))
        print(json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
