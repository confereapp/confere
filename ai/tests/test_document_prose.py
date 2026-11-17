import json
import unittest
from prepare_document_prose import prose_rows, TOPICS


class DocumentProseTests(unittest.TestCase):
    def test_opposite_queries_keep_identical_text_and_change_evidence(self):
        rows=prose_rows('train',10)
        self.assertEqual(rows,prose_rows('train',10))
        self.assertFalse(set(TOPICS['train']) & set(TOPICS['validation']))
        saw_conflict=False
        for first,second in zip(rows[::2],rows[1::2]):
            self.assertEqual(first['units'],second['units'])
            self.assertNotEqual(first['requirement'],second['requirement'])
            for row in [first,second]:
                target=json.loads(row['target'])
                for evidence in target['evidence']:
                    i=int(evidence['unit'].split(':')[1])-1
                    facts=row['scenario']['units'][i]['facts']
                    self.assertIn((row['scenario']['feature'],evidence['status']),facts)
                for i,u in enumerate(row['scenario']['units'],1):
                    if not u['facts']:
                        self.assertFalse(any(e['unit']==f'page:{i}' for e in target['evidence']))
                saw_conflict |= len({e['status'] for e in target['evidence']})==2
        self.assertTrue(saw_conflict)


if __name__=='__main__':unittest.main()
