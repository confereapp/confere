"""Amostra limitada e determinística do viewer; preserva JSON bruto e hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request
import urllib.error
import time

REPO = "Polygl0t/gigaverbo-v2-sft"


def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova; downloads parciais são preservados.")
    def revision(): return json.loads(get("https://huggingface.co/api/datasets/" + REPO))["sha"]
    rev = revision()
    args.output.mkdir(parents=True)
    rng = random.Random(151126)
    jobs = []
    # Limites de linhas conservadores, conforme o card; nunca reutiliza páginas nesta coleta.
    for config, pages, total in [("general", 120, 1235976), ("rewriting", 40, 29150),
                                  ("retrieval", 20, 1977667), ("structured", 20, 163542)]:
        jobs.extend((config, offset * 100) for offset in rng.sample(range(total // 100), pages))
    def fetch(job):
        config, offset = job
        params = dict(dataset=REPO, config=config, split="train", offset=offset, length=100, revision=rev)
        raw = get("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params))
        value = json.loads(raw)
        assert len(value["rows"]) == 100 and not any(r.get("truncated_cells") for r in value["rows"])
        path = args.output / f"{config}-{offset}.json"
        with path.open("xb") as file: file.write(raw)
        return path.name, hashlib.sha256(raw).hexdigest()
    (args.output / "plan.json").write_text(json.dumps(dict(revision=rev, jobs=jobs), indent=2))
    files = {}
    stopped = None
    for job in jobs:
        try:
            name, digest = fetch(job)
            files[name] = digest
        except urllib.error.HTTPError as error:
            if error.code != 429: raise
            stopped = dict(http_status=429, retry_after=error.headers.get("Retry-After"), pending=len(jobs)-len(files))
            break
        time.sleep(1)
    if not files: raise ValueError("Nenhuma página disponível; fonte não utilizável.")
    assert revision() == rev, "Origem mudou durante a coleta; não utilizar download."
    metadata = dict(source="https://huggingface.co/datasets/" + REPO, revision=rev, license="Apache-2.0",
                    method="200 páginas de 100 linhas, amostradas sem reposição por subset, seed 151126; revisão observada antes/depois. Viewer não garante snapshot imutável: JSON e hashes preservados são a referência.",
                    files=files, pages=jobs, stopped=stopped, complete=stopped is None)
    (args.output / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(dict(revision=rev, files=len(files), rows=len(files)*100)), flush=True)


if __name__ == "__main__": main()
