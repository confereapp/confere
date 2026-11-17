import json
import unittest
from evaluate_relation_pilot import shortcut_baseline, summarize


class ShortcutControlTests(unittest.TestCase):
    def test_query_control_ignores_body_and_does_not_fit_test(self):
        def row(query,body,status):
            return dict(requirement=query,units=[dict(text=body)],target=json.dumps(dict(status=status)),evaluation_profile='p')
        train=[row('azul azul','qualquer trecho','atendido'),row('verde verde','qualquer trecho','nao_atendido'),row('branco branco','qualquer trecho','evidencia_insuficiente')]
        test=[row('azul','verde verde','nao_atendido')]
        self.assertEqual(shortcut_baseline(train,test,'query'),['atendido'])
        test[0]['target']=json.dumps(dict(status='atendido'));test[0]['units'][0]['text']='branco'
        self.assertEqual(shortcut_baseline(train,test,'query'),['atendido'])
        self.assertEqual(summarize(test,['nao_atendido'])['p']['correct'],0)


if __name__=='__main__':unittest.main()
