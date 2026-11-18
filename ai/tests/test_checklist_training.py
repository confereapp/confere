import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_checklist_training import row  # noqa: E402
from confere_ai.instruct import load_instructions  # noqa: E402

RIGHTS = "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"


class ChecklistTrainingTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "examples.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            return load_instructions(path)

    def test_rows_from_agreed_annotations_load(self):
        req = "O trabalho deve propor trabalhos futuros."
        self.load([row("doc-C12-pos0", "doc", "train", req, "Como trabalho futuro, propomos ampliar a base.", "atendido", RIGHTS),
                   row("doc-C12-neg", "doc", "train", req, "A base foi coletada em 2020.", "evidencia_insuficiente", RIGHTS),
                   row("val-C12-pos0", "val", "validation", req, "Trabalhos futuros incluem testar em campo.", "atendido", RIGHTS)])

    def test_rejects_test_split_and_altered_text(self):
        good = row("doc-C1-pos0", "doc", "test", "O trabalho deve declarar o objetivo.", "O objetivo é medir X.", "atendido", RIGHTS)
        with self.assertRaises(ValueError): self.load([good])
        altered = row("doc-C1-pos0", "doc", "train", "O trabalho deve declarar o objetivo.", "O objetivo é medir X.", "atendido", RIGHTS)
        altered["units"][0]["text"] = "Outro texto."
        with self.assertRaises(ValueError): self.load([altered])


if __name__ == "__main__": unittest.main()
