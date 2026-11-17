import copy
import json
from pathlib import Path
import unittest

from prepare_paragraph_matrix import matrix_rows, query_ceiling
from probe_conditioned_evidence import sampling_view
from confere_ai.evidence_classifier import paired_indices,unit_weighted_query_pools
import torch


class MatrixTests(unittest.TestCase):
    def source(self):
        return dict(data=[dict(title='D',paragraphs=[dict(context='A norma exige estágio.'),dict(context='A norma trata de bibliotecas.')])])

    def annotations(self):
        return dict(documents=[dict(title='D',split='train',paragraph_indices=[0,1],claims=[
            dict(id='q1',text='O estágio é exigido.',labels=[1,2],rationales=['Afirma explicitamente.','Não trata de estágio.']),
            dict(id='q2',text='O estágio é proibido.',labels=[0,2],rationales=['Exigência incompatível com proibição.','Não trata de estágio.'])])])

    def test_same_query_different_paragraphs_have_different_labels_and_unique_ids(self):
        rows,excluded=matrix_rows(self.source(),self.annotations())
        self.assertEqual(len(rows),4);self.assertEqual(len({r['id'] for r in rows}),4)
        self.assertEqual(excluded,[])
        self.assertEqual(query_ceiling(rows),dict(queries=2,pairs=4,constant_label_queries=0,max_query_only_correct=2))
        self.assertEqual(rows[0]['requirement'],rows[1]['requirement'])
        self.assertNotEqual(rows[0]['target'],rows[1]['target'])
        self.assertEqual(rows[0]['units'][0]['text'],'A norma exige estágio.')

    def test_incomplete_or_constant_matrix_fails_closed(self):
        for labels in [[1],[1,1],[True,2]]:
            annotations=self.annotations();annotations['documents'][0]['claims'][0]['labels']=labels
            with self.assertRaises(ValueError):matrix_rows(self.source(),annotations)

    def test_explicit_exclusion_preserves_its_reason(self):
        annotations=self.annotations();claim=annotations['documents'][0]['claims'][0]
        claim.update(labels=[1,1],exclusion_reason='Não há mudança de relação nos parágrafos disponíveis.')
        rows,excluded=matrix_rows(self.source(),annotations)
        self.assertEqual(len(rows),2);self.assertIn('Não há mudança',excluded[0]['reason'])

    def test_previously_examined_documents_are_not_new_test(self):
        path=Path(__file__).resolve().parents[1]/'data/paragraph-relation-matrix-v2.json'
        annotations=json.loads(path.read_text());test={d['title'] for d in annotations['documents'] if d['split']=='test'}
        self.assertFalse(test & set(annotations['previously_examined_titles']))
        self.assertEqual(test,set(annotations['new_test_titles']))

    def test_sampling_uses_same_query_with_different_body_and_label(self):
        items=[dict(row_id=str(i),unit='page:1',source_key=str(i),labels=[float(i%2),0.]) for i in range(4)]
        lookup={(str(i),'page:1'):(f'consulta-{i//2}',f'trecho-{i}') for i in range(4)}
        view=sampling_view(items,lookup,True)
        indices=paired_indices(view,unit_weighted_query_pools(view),32,torch.Generator().manual_seed(42))
        for a,b in zip(indices[::2],indices[1::2]):
            self.assertEqual(lookup[str(a),'page:1'][0],lookup[str(b),'page:1'][0])
            self.assertNotEqual(lookup[str(a),'page:1'][1],lookup[str(b),'page:1'][1])
            self.assertNotEqual(items[a]['labels'],items[b]['labels'])
        self.assertEqual(items[0]['source_key'],'0')
        self.assertIs(sampling_view(items,lookup,False),items)


if __name__=='__main__':unittest.main()
