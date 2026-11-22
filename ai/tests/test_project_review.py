import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from confere_ai.project_review import batches_for, input_ids, review_extracted, RESPONSE_TOKENS
from confere_ai.tokenizer import ByteBPE


class ProjectReviewTests(unittest.TestCase):
    def test_unit_predictor_preserves_coverage_quotes_and_conflicts(self):
        doc = self.doc("Há testes. Não há testes.")
        seen = []
        def predict(requirement, units):
            from confere_ai.evidence_classifier import response
            seen.append((requirement, units))
            return response(units, [[.9, .9] for _ in units])
        result = review_extracted([doc], ["Testes"], ByteBPE([]), 512, None, 2, 64, predict_units=predict)["requirements"][0]
        self.assertTrue(result["conflicting_evidence"])
        self.assertEqual(result["status"], "evidencia_insuficiente")
        self.assertEqual(seen[0][0], "Testes")
        self.assertEqual(seen[0][1][0]["text"], doc["units"][0]["text"])
        self.assertTrue(all(e["quote"] == doc["units"][0]["text"] for e in result["evidence"]))
        self.assertEqual(result["coverage"]["processed_characters"], len(doc["units"][0]["text"]))
        self.assertEqual(result["calls"][0]["inference_kind"], "unit_classifier")
        self.assertNotIn("input_tokens", result["calls"][0])
        with self.assertRaises(ValueError):
            review_extracted([doc], ["Testes"], ByteBPE([]), 512, lambda ids:"", 2, 64, predict_units=predict)

    def test_top_k_retrieval_ignores_irrelevant_pages_and_missing_text(self):
        from confere_ai.evidence_classifier import response
        doc = dict(name="a.pdf", sha256="x", format="pdf", warnings=[], units=[
            dict(id="page:1", text="Folha de rosto da universidade."),
            dict(id="page:2", text="O sistema usa autenticação de dois fatores."),
            dict(id="page:3", text=""),
            dict(id="page:4", text="Sumário e agradecimentos.")])
        seen = []
        def predict(requirement, units):
            seen.extend(u["text"] for u in units)
            # Classificador ruidoso: marca oposto em tudo que não cita autenticação.
            return response(units, [[.9, .1] if "autentica" in u["text"] else [.1, .9] for u in units])
        row = review_extracted([doc], ["Autenticação de dois fatores"], ByteBPE([]), 512, None, 2, 64,
                               predict_units=predict, top_k=1)["requirements"][0]
        self.assertEqual(seen, ["O sistema usa autenticação de dois fatores."])
        self.assertEqual(row["status"], "atendido")
        self.assertEqual(row["coverage"]["units_without_text"], [dict(document_id="doc:1", unit="page:3")])
        self.assertFalse(row["coverage"]["all_extracted_text_processed"])
        tie = review_extracted([doc], ["Autenticação de dois fatores"], ByteBPE([]), 512, None, 2, 64,
                               predict_units=predict, top_k=2)["requirements"][0]
        self.assertEqual(tie["status"], "evidencia_insuficiente")
        self.assertTrue(tie["conflicting_evidence"])

    def test_checkpoint_loader_accepts_classifier_without_decoder_and_records_real_kind(self):
        from dataclasses import asdict
        from unittest.mock import patch
        import torch
        from confere_ai.evidence_classifier import EvidenceClassifier
        from confere_ai.model import GPT, GPTConfig
        from confere_ai.project_review import review_documents
        config = GPTConfig(vocab_size=260, block_size=512, width=16, layers=1, heads=2)
        model = EvidenceClassifier(GPT(config))
        with torch.no_grad():
            model.head.weight.zero_(); model.head.bias.copy_(torch.tensor([9., -9.]))
        checkpoint = dict(config=asdict(config), merges=[], classifier=model.state_dict(), step=7,
                          settings=dict(objective="document_classifier", readout="shared_json_decision_prefix_v1"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"classifier.pt"; torch.save(checkpoint, path)
            with patch.object(GPT, "generate", side_effect=AssertionError("Classifier cannot generate")):
                result = review_documents([self.doc("Há testes.")], ["Testes"], path, "cpu")
        self.assertEqual(result["requirements"][0]["status"], "atendido")
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["model"]["inference_kind"], "unit_classifier")
        self.assertNotIn("response_tokens", result["model"])
        self.assertEqual(result["model"]["fragment_token_reserve"], 256)

    def test_pilot_audit_rejects_coverage_holes_and_modified_quotes(self):
        spec=importlib.util.spec_from_file_location("full_files",Path(__file__).parents[1]/"scripts/evaluate_full_files.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        docs=[self.doc("O projeto inclui testes.")]
        result=review_extracted(docs,["Testes"],ByteBPE([]),256,lambda ids:'{"status":"atendido","unit":"page:1"}')
        self.assertTrue(module.audit_result(result,docs)["contiguous_text_coverage_verified"])
        result["requirements"][0]["calls"][0]["sources"][0]["char_end"]-=1
        with self.assertRaises(AssertionError):module.audit_result(result,docs)
        result["requirements"][0]["calls"][0]["sources"][0]["char_end"]+=1
        result["requirements"][0]["evidence"][0]["quote"]="Texto inventado"
        with self.assertRaises(AssertionError):module.audit_result(result,docs)

    def test_local_alias_maps_to_correct_file_even_with_repeated_basename(self):
        docs=[self.doc("O projeto não inclui testes.",digest="a"),self.doc("O projeto inclui testes.",digest="b")]
        docs[1]["units"][0]["id"]="table:2:row:3"
        row=review_extracted(docs,["Testes"],ByteBPE([]),512,lambda ids:'{"status":"atendido","unit":"page:2"}')["requirements"][0]
        self.assertEqual(row["status"],"atendido")
        self.assertEqual(row["evidence"][0]["document_id"],"doc:2")
        self.assertEqual(row["evidence"][0]["source_unit"],"table:2:row:3")
        self.assertEqual(row["evidence"][0]["source_sha256"],"b")
        self.assertEqual(row["evidence"][0]["quote"],docs[1]["units"][0]["text"])

    def doc(self,text,name="project.docx",digest="a"):
        return dict(name=name,sha256=digest,format="docx",warnings=[],units=[dict(id="paragraph:9",text=text)])

    def test_long_unicode_units_have_exact_contiguous_coverage_and_fit_context(self):
        tokenizer=ByteBPE([])
        documents=[self.doc("á🙂 evidência com espaços. "*60),self.doc("Outro documento. "*30,"project.docx","b")]
        batches=batches_for(documents,"Incluir testes.",tokenizer,256)
        self.assertGreater(len(batches),2)
        by_source=defaultdict(list)
        for batch in batches:
            self.assertLessEqual(len(batch),4)
            aliases=[dict(id=f"page:{i+1}",text=p["text"]) for i,p in enumerate(batch)]
            self.assertLessEqual(len(input_ids("Incluir testes.",aliases,tokenizer))+RESPONSE_TOKENS,256)
            for part in batch:by_source[part["document_id"]].append(part)
        for i,doc in enumerate(documents,1):
            parts=by_source[f"doc:{i}"]
            self.assertEqual(parts[0]["char_start"],0)
            self.assertEqual(parts[-1]["char_end"],len(doc["units"][0]["text"]))
            self.assertTrue(all(a["char_end"]==b["char_start"] for a,b in zip(parts,parts[1:])))
            self.assertEqual("".join(p["text"] for p in parts),doc["units"][0]["text"])

    def test_conflicting_chunks_are_inconclusive_and_quotes_keep_original_offsets(self):
        doc=self.doc("O projeto inclui testes. "*45)
        raw=iter([json.dumps(dict(status=s,unit="page:1")) for s in ("atendido","nao_atendido")])
        calls=[]
        def predict(ids):
            calls.append(ids)
            return next(raw,'{"status":"evidencia_insuficiente","unit":null}')
        result=review_extracted([doc],["Incluir testes."],ByteBPE([]),256,predict)["requirements"][0]
        self.assertTrue(result["conflicting_evidence"])
        self.assertEqual(result["status"],"evidencia_insuficiente")
        self.assertEqual(len(calls),result["coverage"]["batches"])
        for evidence in result["evidence"]:
            self.assertEqual(evidence["source_unit"],"paragraph:9")
            self.assertEqual(evidence["quote"],doc["units"][0]["text"][evidence["char_start"]:evidence["char_end"]])

    def test_invalid_reference_and_missing_text_prevent_conclusive_status(self):
        doc=self.doc("O projeto inclui testes.")
        for raw in ('{"status":"atendido","unit":"page:4"}',"texto inválido"):
            result=review_extracted([doc],["Testes"],ByteBPE([]),256,lambda ids:raw)["requirements"][0]
            self.assertEqual(result["status"],"evidencia_insuficiente")
            self.assertFalse(result["model_outputs_valid"])
        doc["units"].append(dict(id="page:2",text=""))
        result=review_extracted([doc],["Testes"],ByteBPE([]),256,lambda ids:'{"status":"atendido","unit":"page:1"}')["requirements"][0]
        self.assertEqual(result["status"],"evidencia_insuficiente")
        self.assertTrue(result["coverage"]["units_without_text"])

    def test_requirement_budget_is_checked_before_any_prediction(self):
        calls=[]
        with self.assertRaisesRegex(ValueError,"requisito"):
            review_extracted([self.doc("Texto")],["Testes","a"*1000],ByteBPE([]),256,lambda ids:calls.append(ids))
        self.assertFalse(calls)

    def test_cli_rejects_overwrite_before_loading_files_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"result.json";output.write_text("preservado")
            result=subprocess.run([sys.executable,"-m","confere_ai.cli","review-project","missing.pdf","--requirement","Testes","--checkpoint","missing.pt","--output",str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("saída já existe",result.stderr)
            self.assertEqual(output.read_text(),"preservado")
