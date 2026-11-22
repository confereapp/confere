import json
import unittest

from confere_ai.chat_api import converse, parse_request
from confere_ai.chat import prepare_prompt
from confere_ai.tokenizer import ByteBPE


class ChatApiTests(unittest.TestCase):
    def test_boundary_rejects_invalid_or_unbounded_requests(self):
        self.assertEqual(parse_request(b'{"message":"oi"}'), dict(message="oi", history=[]))
        self.assertEqual(parse_request(b'{"message":"oi","history":[{"message":"oi","response":""}]}')["history"][0]["response"], "")
        for value in [[], {}, dict(message=" "), dict(message="a" * 2001),
                      dict(message="oi", checkpoint="private"), dict(message="oi", history=[{}]),
                      dict(message="oi", history=[dict(message="oi", response="sim")] * 9)]:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                parse_request(json.dumps(value).encode())
        with self.assertRaises(ValueError):
            parse_request(b" " * 200001)

    def test_conversation_uses_terminal_prompt_and_reports_dropped_turns(self):
        request = dict(message="oi", history=[dict(message="antiga", response="resposta"),
                                             dict(message="nova", response="sim")])
        prompts = []
        def respond(ids):
            prompts.append(ids)
            return "olá 🙂"
        for objective in ["conversation", "instruction", "causal_lm"]:
            result = converse(request, ByteBPE(), respond, 24, 10, objective)
            expected, dropped = prepare_prompt([("antiga", "resposta"), ("nova", "sim")],
                                              "oi", ByteBPE(), 24, 10, objective)
            self.assertEqual(prompts[-1], expected)
            self.assertEqual(result, dict(response="olá 🙂", experimental=True, dropped_turns=dropped))

    def test_long_current_message_does_not_run_generation(self):
        def fail(_):
            self.fail("Generation must not run")
        result = converse(dict(message="a" * 50, history=[]), ByteBPE(), fail, 32, 10, "instruction")
        self.assertEqual(result, {"error": "context_length"})
