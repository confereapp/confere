import copy
import unittest
from evaluate_real_documents import audit
from confere_ai.project_review import review_extracted
from confere_ai.tokenizer import ByteBPE


class RealDocumentAuditTests(unittest.TestCase):
    def test_coverage_quotes_and_inconclusive_status_are_distinct(self):
        doc=dict(name='source.pdf',sha256='x',format='pdf',warnings=[],
                 units=[dict(id='page:1',text='Há testes.'),dict(id='page:2',text='')])
        case=dict(id='real-1',requirement=dict(text='Testes'),
                  expected=dict(status='atendido',evidence=[dict(unit='page:1',quote='Há testes.')]))
        result=review_extracted([doc],['Testes'],ByteBPE([]),512,
                               lambda ids:'{"status":"atendido","evidence":[{"status":"atendido","unit":"page:1"}]}',2,64)
        row=audit(result,[doc],[case])[0]
        self.assertFalse(row['status_match'])
        self.assertEqual(row['annotated_quote_hits'],1)
        self.assertTrue(row['missing_text_units'])
        for kind in ['quote','coverage']:
            bad=copy.deepcopy(result)
            if kind=='quote':bad['requirements'][0]['evidence'][0]['quote']='Inventado'
            else:bad['requirements'][0]['calls'][0]['sources'][0]['char_end']-=1
            with self.assertRaises(AssertionError):audit(bad,[doc],[case])


if __name__=='__main__':unittest.main()
