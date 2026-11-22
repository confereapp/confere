import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_intent_replay import build
from confere_ai.instruct import load_instructions


class IntentReplayTests(unittest.TestCase):
    def test_continuations_use_authored_targets_and_keep_reserved_rows_unchanged(self):
        path = Path(__file__).resolve().parents[1] / "data/authored/conversations.jsonl"
        authored = [json.loads(line) for line in path.read_text().splitlines()]
        source = [dict(r, response_tokens=256) for r in authored if r["split"] != "test"]
        rows = build(source, authored, 80)
        self.assertEqual(rows, build(source, authored, 80))
        self.assertEqual([r for r in rows if r["split"] != "train"], [r for r in source if r["split"] != "train"])
        added = rows[len(source):]
        definitions = [r for r in added if r["task"] == "conversation_definition" and r["turn_index"] == 1]
        self.assertTrue(definitions)
        pairs = {(r["history"][-1][1], r["target"]) for r in authored if r["split"] == "train"
                 and r["task"] == "conversation_definition" and r["turn_index"] == 1}
        self.assertTrue(all((r["history"][-1][1], r["target"]) in pairs for r in definitions))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "rows.jsonl"
            output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            self.assertEqual(len(load_instructions(output)[0]), len(rows))
        broken = [r for r in authored if r["task"] != "conversation_definition"]
        with self.assertRaises(ValueError): build(source, broken, 8)
