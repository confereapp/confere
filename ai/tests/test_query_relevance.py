import json
import unittest
from probe_query_relevance import candidate, predict


class QueryRelevanceTests(unittest.TestCase):
    def test_filter_keeps_negative_and_accent_variations_but_rejects_unrelated(self):
        q='O relatório deve apresentar notificações.'
        self.assertTrue(candidate(q,'Não existem notificações na entrega.'))
        self.assertTrue(candidate(q,'A entrega fornece notificacoes.'))
        self.assertFalse(candidate(q,'A equipe escolheu cores para a interface.'))
        self.assertTrue(candidate('O relatório deve apresentar.','Texto qualquer.'))

    def test_filter_does_not_invent_positive_decision_or_remove_opposite_evidence(self):
        class Predictor:
            thresholds=[.5,.5]
            def probabilities(self,q,units):return [[.1,.9] for u in units]
        raw=json.loads(predict(Predictor(),'O projeto deve incluir backup.',[
            dict(id='page:1',text='Não há backup.'),dict(id='page:2',text='Cores e interface.')]))
        self.assertEqual(raw,dict(status='nao_atendido',evidence=[dict(status='nao_atendido',unit='page:1')]))
        raw=json.loads(predict(Predictor(),'O projeto deve incluir backup.',[dict(id='page:1',text='Cores e interface.')]))
        self.assertEqual(raw,dict(status='evidencia_insuficiente',evidence=[]))


if __name__=='__main__':unittest.main()
