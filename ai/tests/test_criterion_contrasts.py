import json
import unittest
from prepare_criterion_contrasts import criterion_rows


class CriterionContrastTests(unittest.TestCase):
    def test_unavailability_does_not_reverse_explicit_component_inclusion(self):
        checked=0
        for row in criterion_rows('train',100):
            target=json.loads(row['target'])
            for i,part in enumerate(row['scenario']['parts'],1):
                if part['topic']==row['scenario']['topic'] and 'temporariamente indisponível' in part['text']:
                    expected='nao_atendido' if row['scenario']['criterion']=='absence' else 'atendido'
                    self.assertIn(dict(status=expected,unit=f'page:{i}'),target['evidence'])
                    checked+=1
        self.assertGreater(checked,0)

    def test_legacy_unscoped_generated_criteria_are_rejected(self):
        import tempfile
        from pathlib import Path
        from confere_ai.instruct import load_instructions
        rows=criterion_rows('train',5)+criterion_rows('validation',5)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'examples.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(len(load_instructions(path)[0]),len(rows))
            del rows[0]['provenance']['annotation_scope']
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Escopo de critério'):load_instructions(path)

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
