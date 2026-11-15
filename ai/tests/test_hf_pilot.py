import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_hf_pilot import public_rows


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
