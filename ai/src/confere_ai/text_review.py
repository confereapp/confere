"""Ponte JSON por stdin para análise textual experimental da API local."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .project_review import review_documents


def parse_request(raw):
    if len(raw) > 50000:
        raise ValueError("Entrada excede 50 KB.")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"text", "requirements"}:
        raise ValueError("Informe text e requirements.")
    text, requirements = value["text"], value["requirements"]
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("Texto deve ter de 1 a 8000 caracteres.")
    if (not isinstance(requirements, list) or not 1 <= len(requirements) <= 4
        or any(not isinstance(r, str) or not r.strip() or len(r) > 600 for r in requirements)):
        raise ValueError("Informe de 1 a 4 requisitos de até 600 caracteres.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    try:
        request = parse_request(sys.stdin.buffer.read(50001))
        text = request["text"]
        document = dict(name="submission.txt", sha256=hashlib.sha256(text.encode()).hexdigest(),
                        format="text", warnings=["Entrada textual; não verifica conteúdo de anexos."],
                        units=[dict(id="text:1", text=text)])
        result = review_documents([document], request["requirements"], args.checkpoint, args.device)
        # Caminhos locais não fazem parte do contrato público da API.
        result["model"].pop("checkpoint", None)
        print(json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
