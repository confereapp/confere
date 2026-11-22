import json
import unittest
from confere_ai.grounding import resolve, judge, prompt_for


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.units = [{"id":"page:1", "text":"O projeto inclui testes."}, {"id":"page:2", "text":"Tema: biblioteca."}]

    def test_quote_is_from_source_and_extra_model_quote_is_rejected(self):
        raw=json.dumps({"status":"atendido","unit":"page:1"})
        result=resolve(raw,"projeto.pdf",self.units)
        self.assertEqual(result["result"]["evidence"][0]["quote"],self.units[0]["text"])
        bad=json.dumps({"status":"atendido","unit":"page:1","quote":"inventada"})
        self.assertFalse(resolve(bad,"projeto.pdf",self.units)["valid_format"])

    def test_unknown_or_ambiguous_reference_fails_closed(self):
        for units,unit in ((self.units,"page:3"),(self.units+self.units,"page:1")):
            result=resolve(json.dumps({"status":"atendido","unit":unit}),"projeto.pdf",units)
            self.assertFalse(result["reference_valid"])
            self.assertEqual(result["result"]["status"],"evidencia_insuficiente")
            self.assertEqual(result["result"]["evidence"],[])

    def test_wrong_status_is_not_counted_as_correct_even_with_real_quote(self):
        expected=json.dumps({"status":"nao_atendido","unit":"page:1"})
        row={"document":"projeto.pdf","units":self.units,"target":expected}
        result=judge(json.dumps({"status":"atendido","unit":"page:1"}),row)
        self.assertTrue(result["reference_valid"])
        self.assertFalse(result["status_correct"])
        self.assertFalse(result["exact_match"])

    def test_invalid_outputs_and_null_evidence_are_checked(self):
        for raw in ('[]','null','texto','{"status":"atendido","unit":null}', '{"status":"evidencia_insuficiente","unit":"page:1"}', '{"status":"atendido","status":"nao_atendido","unit":"page:1"}'):
            self.assertFalse(resolve(raw,"p.pdf",self.units)["valid_format"])
        result=resolve('{"status":"evidencia_insuficiente","unit":null}',"p.pdf",self.units)
        self.assertTrue(result["reference_valid"])
        self.assertEqual(result["result"]["evidence"],[])

    def test_prompt_retains_unit_identity_and_text(self):
        prompt=prompt_for("Incluir testes",self.units)
        self.assertIn("[page:1] O projeto inclui testes.",prompt)


class MultipleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.units = [{"id": "page:1", "text": "Inclui testes."},
                      {"id": "page:2", "text": "Não inclui testes."}]
        self.value = dict(status="evidencia_insuficiente", evidence=[
            dict(status="atendido", unit="page:1"), dict(status="nao_atendido", unit="page:2")])

    def test_conflict_keeps_both_source_quotes_and_statuses(self):
        result = resolve(json.dumps(self.value), "entrega", self.units)
        self.assertTrue(result["reference_valid"])
        self.assertEqual([e["quote"] for e in result["result"]["evidence"]], [u["text"] for u in self.units])
        row = dict(document="entrega", units=self.units, target=json.dumps(self.value))
        reordered = dict(self.value, evidence=list(reversed(self.value["evidence"])))
        self.assertTrue(judge(json.dumps(reordered), row)["exact_match"])
        missing = dict(status="atendido", evidence=self.value["evidence"][:1])
        self.assertFalse(judge(json.dumps(missing), row)["exact_match"])
        row["grounding_schema"] = 2
        legacy = json.dumps(dict(status="evidencia_insuficiente", unit=None))
        self.assertFalse(judge(legacy, row)["valid_format"])

    def test_malformed_conflicts_duplicates_and_unknown_sources_fail_closed(self):
        bad = [dict(self.value, status="atendido"), dict(status="atendido", evidence=[]),
               dict(self.value, evidence=self.value["evidence"] * 2),
               dict(status="atendido", evidence=[dict(status="atendido", unit="page:9")]),
               dict(status="atendido", evidence=[dict(status="atendido", unit="page:1", quote="fake")])]
        for value in bad:
            result = resolve(json.dumps(value), "entrega", self.units)
            self.assertFalse(result["valid_format"] and result["reference_valid"])
            self.assertEqual(result["result"]["evidence"], [])

    def test_same_batch_conflict_is_inconclusive_with_exact_offsets(self):
        from confere_ai.project_review import review_extracted
        from confere_ai.tokenizer import ByteBPE
        doc = dict(name="entrega", sha256="a", format="text", warnings=[], units=self.units)
        result = review_extracted([doc], ["Testes"], ByteBPE([]), 1024,
                                  lambda ids: json.dumps(self.value), schema_version=2, response_tokens=192)
        row = result["requirements"][0]
        self.assertTrue(row["conflicting_evidence"])
        self.assertTrue(row["model_outputs_valid"])
        self.assertEqual(row["status"], "evidencia_insuficiente")
        self.assertEqual(len(row["evidence"]), 2)
        self.assertEqual({e["model_status"] for e in row["evidence"]}, {"atendido", "nao_atendido"})
