import json
import unittest

from confere_ai.text_review import parse_request


class TextReviewTests(unittest.TestCase):
    def test_preserves_unicode_and_original_offsets(self):
        request = dict(text="  ação 🙂  ", requirements=["Incluir testes."])
        self.assertEqual(parse_request(json.dumps(request).encode()), request)

    def test_rejects_invalid_or_unbounded_inputs_before_loading_model(self):
        for value in ([], {}, dict(text=" ", requirements=["Testes"]),
                      dict(text="a"*8001, requirements=["Testes"]),
                      dict(text="Texto", requirements=[]), dict(text="Texto", requirements=[True]),
                      dict(text="Texto", requirements=[" "]), dict(text="Texto", requirements=["a"*601]),
                      dict(text="Texto", requirements=["Testes"]*5),
                      dict(text="Texto", requirements=["Testes"], checkpoint="other.pt")):
            with self.assertRaises(ValueError):
                parse_request(json.dumps(value).encode())
        with self.assertRaises(ValueError):
            parse_request(b" "*50001)
