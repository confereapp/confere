"""Ponte JSON por stdin para conferir um arquivo de entrega (PDF/DOCX) contra requisitos."""
import argparse
import json
import sys
from pathlib import Path

from .project_review import review_files

# Os 93% do teste cego foram medidos com top_k=8; mudar aqui invalida a métrica.
TOP_K = 8


def parse_request(raw):
    if len(raw) > 200000:
        raise ValueError("Entrada excede 200 KB.")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError("Informe path e requirements.")
    requirements = value.get("requirements")
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 100:
        raise ValueError("Informe de 1 a 100 requisitos.")
    for item in requirements:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not isinstance(item.get("title"), str) or not item["title"].strip()
                or not isinstance(item.get("description") or "", str)):
            raise ValueError("Cada requisito precisa de id e title.")
    return value


def requirement_text(item):
    description = (item.get("description") or "").strip()
    return f"{item['title'].strip()}: {description}" if description else item["title"].strip()


def summarize(result, requirements):
    """Reduz o resultado ao contrato da API: status e evidências por requisito, na ordem recebida."""
    return dict(
        requirements=[dict(id=item["id"], status=r["status"],
                           evidence=[dict(source_unit=e["source_unit"], quote=e["quote"],
                                          model_status=e["model_status"]) for e in r["evidence"]])
                      for item, r in zip(requirements, result["requirements"])],
        warnings=[w for s in result["sources"] for w in s["warnings"]],
        model=dict(sha256=result["model"]["sha256"], step=result["model"]["step"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    try:
        request = parse_request(sys.stdin.buffer.read(200001))
        requirements = request["requirements"]
        result = review_files([Path(request["path"])], [requirement_text(r) for r in requirements],
                              args.checkpoint, args.device, top_k=TOP_K)
        print(json.dumps(summarize(result, requirements), ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
