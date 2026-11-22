import unittest

from confere_ai.feedback import explain


def ev(unit, status="atendido"):
    return dict(source_unit=unit, quote="x", model_status=status)


class FeedbackTests(unittest.TestCase):
    def test_met_lists_supporting_places_once(self):
        text = explain("atendido", [ev("page:4"), ev("page:4"), ev("page:9"), ev("page:2", "evidencia_insuficiente")])
        self.assertIn("3 trechos sustentam", text)
        self.assertIn("(p. 4 e p. 9)", text)

    def test_single_paragraph(self):
        self.assertIn("1 trecho sustenta o requisito (parágrafo 7)", explain("atendido", [ev("paragraph:7")]))

    def test_never_claims_missing(self):
        for status in ("evidencia_insuficiente", "nao_atendido", "atendido"):
            text = explain(status, [])
            self.assertIn("revise manualmente", text)
            self.assertNotIn("falta.", text)

    def test_contradiction_is_not_rejection(self):
        self.assertIn("Não é reprovação", explain("nao_atendido", [ev("page:3", "nao_atendido")]))


if __name__ == "__main__":
    unittest.main()
