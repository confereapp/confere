import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_hf_pilot import fits, public_rows
from confere_ai.tokenizer import ByteBPE


class HfPilotTests(unittest.TestCase):
    def test_system_instructions_and_broken_roles_are_not_silently_removed(self):
        for messages in [[dict(role="system", content="regra"), dict(role="user", content="oi")],
                         [dict(role="user", content="oi")],
                         [dict(role="user", content="oi"), dict(role="assistant", content=" ")]]:
            self.assertEqual(public_rows(messages, "revision"), [])

    def test_complete_conversation_stays_in_one_split_and_keeps_history(self):
        messages = [dict(role=role, content=text) for role, text in
                    [("user", "oi"), ("assistant", "olá"), ("user", "como vai?"), ("assistant", "bem")]]
        rows = public_rows(messages, "revision")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["history"], [["oi", "olá"]])
        self.assertEqual(rows[0]["project_id"], rows[1]["project_id"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertEqual(rows, public_rows(messages, "revision"))


    def test_answer_budget_excludes_without_truncating(self):
        row = dict(id="short", prompt="oi", target="olá 🙂", history=[], response_tokens=256)
        tokenizer = ByteBPE()
        budget = len(tokenizer.encode(row["target"])) + 1
        self.assertTrue(fits(row, tokenizer, 1024, budget))
        self.assertFalse(fits(row, tokenizer, 1024, budget-1))
        self.assertEqual(row["target"], "olá 🙂")
        self.assertFalse(fits({**row, "prompt": "a"*1024}, tokenizer, 1024, budget))


class HfCollectionTests(unittest.TestCase):
    def test_rate_limit_stops_requests_and_marks_partial_snapshot(self):
        import contextlib
        import io
        import json
        import tempfile
        import urllib.error
        from unittest.mock import patch
        import fetch_hf_sample
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sample"
            calls = []
            def get(url):
                calls.append(url)
                if "/api/datasets/" in url:
                    return json.dumps({"sha": "revision"}).encode()
                if sum("/rows?" in item for item in calls) == 1:
                    return json.dumps({"rows": [{"row": {}} for _ in range(100)]}).encode()
                raise urllib.error.HTTPError(url, 429, "rate limit", {"Retry-After": "60"}, None)
            with patch.object(sys, "argv", ["fetch_hf_sample", str(output)]), patch.object(fetch_hf_sample, "get", side_effect=get), patch.object(fetch_hf_sample.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                fetch_hf_sample.main()
            metadata = json.loads((output / "source.json").read_text())
            self.assertFalse(metadata["complete"])
            self.assertEqual(metadata["stopped"]["http_status"], 429)
            self.assertEqual(len(metadata["files"]), 1)
            self.assertEqual(sum("/rows?" in item for item in calls), 2)
