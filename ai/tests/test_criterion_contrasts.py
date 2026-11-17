import json
import unittest
from prepare_criterion_contrasts import criterion_rows


class CriterionContrastTests(unittest.TestCase):
    def test_absence_is_positive_for_absence_criterion_and_mention(self):
        rows=criterion_rows('train',5)
        cases={r['scenario']['criterion']:r for r in rows if r['project_id']=='criterion-train-1'
               and any(p['topic']==r['scenario']['topic'] and p['state']=='nao_atendido' for p in r['scenario']['parts'])}
        self.assertEqual({k:json.loads(r['target'])['status'] for k,r in cases.items()},
                         dict(presence='nao_atendido',absence='atendido',mention='atendido'))
        self.assertEqual(cases['presence']['units'],cases['absence']['units'])
        self.assertEqual(cases['presence']['units'],cases['mention']['units'])

    def test_opposite_facts_do_not_conflict_for_a_mention_criterion(self):
        rows=criterion_rows('train',5)
        cases=[r for r in rows if r['project_id']=='criterion-train-2'
               and len([p for p in r['scenario']['parts'] if p['topic']==r['scenario']['topic']])==2]
        self.assertEqual(len(cases),3)
        for r in cases:
            target=json.loads(r['target'])
            self.assertEqual(target['status'],'atendido' if r['scenario']['criterion']=='mention' else 'evidencia_insuficiente')
            self.assertEqual(len(target['evidence']),2)


if __name__=='__main__':unittest.main()
