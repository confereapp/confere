import json
import unittest
from evaluate_relation_pilot import shortcut_baseline, summarize, query_groups,require_validation_gate


class ShortcutControlTests(unittest.TestCase):
    def test_fresh_matrix_test_is_blocked_before_validation_passes(self):
        rows=[dict(provenance=dict(relation_matrix_version=2))]
        for summary in [{},dict(heldout_eligible=False),dict(heldout_eligible=1)]:
            with self.assertRaisesRegex(ValueError,'Reserva da matriz preservada'):require_validation_gate(rows,summary)
        require_validation_gate(rows,dict(heldout_eligible=True))
        require_validation_gate([dict(provenance={})],{})

    def test_query_control_ignores_body_and_does_not_fit_test(self):
        def row(query,body,status):
            return dict(requirement=query,units=[dict(text=body)],target=json.dumps(dict(status=status)),evaluation_profile='p')
        train=[row('azul azul','qualquer trecho','atendido'),row('verde verde','qualquer trecho','nao_atendido'),row('branco branco','qualquer trecho','evidencia_insuficiente')]
        test=[row('azul','verde verde','nao_atendido')]
        self.assertEqual(shortcut_baseline(train,test,'query'),['atendido'])
        test[0]['target']=json.dumps(dict(status='atendido'));test[0]['units'][0]['text']='branco'
        self.assertEqual(shortcut_baseline(train,test,'query'),['atendido'])
        self.assertEqual(summarize(test,['nao_atendido'])['p']['correct'],0)

    def test_matching_unknown_status_with_wrong_evidence_is_not_correct(self):
        row=dict(target=json.dumps(dict(status='evidencia_insuficiente')),evaluation_profile='p')
        self.assertEqual(summarize([row],['evidencia_insuficiente'],[False])['p']['correct'],0)
        with self.assertRaises(ValueError):summarize([row],[],[])

    def test_fixed_query_requires_all_contexts_and_evidence_correct(self):
        rows=[dict(id=str(i),source_document='D',requirement='Q',provenance=dict(kind='public_paragraph_criterion'),target=str(i)) for i in range(2)]
        samples=[dict(id='0',exact_match=True),dict(id='1',exact_match=False)]
        result=query_groups(rows,samples)
        self.assertEqual((result['informative'],result['all_pairs_correct']),(1,0))
        samples[1]['exact_match']=True
        self.assertEqual(query_groups(rows,samples)['all_pairs_correct'],1)


if __name__=='__main__':unittest.main()
