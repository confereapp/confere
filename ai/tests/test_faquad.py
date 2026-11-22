import copy
import unittest

from prepare_faquad import public_rows
from confere_ai.qa_source import validate_provenance


class FaquadTests(unittest.TestCase):
    def test_titles_and_contexts_are_reserved_and_offsets_verified(self):
        data={'data':[dict(title=str(i),paragraphs=[dict(context='Texto '+str(i),qas=[dict(
            id=str(i),question='Qual texto?',answers=[dict(text=str(i),answer_start=6)])])]) for i in range(10)]}
        rows=list(public_rows(data))
        for row in rows: validate_provenance(row)
        for field in ('title','context'):
            train={r['provenance'][field] for r in rows if r['split']=='train'}
            val={r['provenance'][field] for r in rows if r['split']=='validation'}
            self.assertTrue(train and val)
            self.assertFalse(train & val)
        for field,value in [('answer_start',-1),('license','MIT'),('context','alterado'),('revision','main')]:
            bad=copy.deepcopy(rows[0]);bad['provenance'][field]=value
            with self.assertRaises(ValueError): validate_provenance(bad)
        bad=copy.deepcopy(rows[0]);bad['prompt']+=' outro texto'
        with self.assertRaises(ValueError): validate_provenance(bad)


if __name__=='__main__': unittest.main()
