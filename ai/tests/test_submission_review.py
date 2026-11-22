import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from confere_ai import submission_review


class SubmissionReviewTests(unittest.TestCase):
    def test_rejects_invalid_requests_before_loading_model(self):
        for value in ([], {}, dict(path="a.pdf"), dict(path="a.pdf", requirements=[]),
                      dict(path="a.pdf", requirements=[dict(id="r1")]),
                      dict(path="a.pdf", requirements=[dict(id="r1", title=" ")]),
                      dict(path=1, requirements=[dict(id="r1", title="Testes")])):
            with self.assertRaises(ValueError):
                submission_review.parse_request(json.dumps(value).encode())

    def test_uses_top_k_8_and_keeps_requirement_ids(self):
        result = dict(
            sources=[dict(warnings=["aviso"])], model=dict(sha256="a"*64, step=2000, checkpoint="/x"),
            requirements=[
                dict(status="atendido", evidence=[dict(source_unit="page:3", quote="Há testes.", model_status="atendido",
                                                       char_start=0, char_end=10)]),
                dict(status="evidencia_insuficiente", evidence=[]),
            ])
        request = dict(path="/tmp/entrega.pdf", requirements=[
            dict(id="r1", title="Testes", description="unitários"), dict(id="r2", title="Diagrama")])
        stdin = mock.Mock(buffer=io.BytesIO(json.dumps(request).encode()))
        out = io.StringIO()
        with mock.patch.object(submission_review, "review_files", return_value=result) as review, \
                mock.patch("sys.argv", ["x", "--checkpoint", "best.pt", "--device", "cpu"]), \
                mock.patch("sys.stdin", stdin), redirect_stdout(out):
            self.assertEqual(submission_review.main(), 0)
        paths, texts, _, device = review.call_args.args
        self.assertEqual(review.call_args.kwargs, dict(top_k=8))
        self.assertEqual((texts, device), (["Testes: unitários", "Diagrama"], "cpu"))
        self.assertEqual(json.loads(out.getvalue()), dict(
            requirements=[dict(id="r1", status="atendido",
                               evidence=[dict(source_unit="page:3", quote="Há testes.", model_status="atendido")]),
                          dict(id="r2", status="evidencia_insuficiente", evidence=[])],
            warnings=["aviso"], model=dict(sha256="a"*64, step=2000)))


if __name__ == "__main__":
    unittest.main()
