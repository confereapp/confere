import copy
import unittest
from confere_ai.generative_selection import quality_key


class GenerativeSelectionTests(unittest.TestCase):
    def test_worst_family_matters_more_than_total_and_context_requires_ninety_percent(self):
        rows = [dict(task="conversation_definition", exact_match=i == 0) for i in range(4)]
        rows += [dict(task="conversation_context_answer_fact", exact_match=True) for _ in range(12)]
        old = quality_key(dict(samples=rows))
        improved = copy.deepcopy(rows)
        improved[1]["exact_match"] = True
        improved[-1]["exact_match"] = False
        self.assertGreater(quality_key(dict(samples=improved)), old)
        all_correct = [dict(r, exact_match=True) for r in rows]
        self.assertTrue(quality_key(dict(samples=all_correct))[0])
        all_correct[-1]["exact_match"] = all_correct[-2]["exact_match"] = False
        self.assertFalse(quality_key(dict(samples=all_correct))[0])
        doc = dict(task="grounded_atendido", exact_match=True, evaluation_profile="positive",
                   valid_format=True, reference_valid=True, result=dict(status="atendido"), expected='{"status":"atendido"}')
        self.assertTrue(quality_key(dict(samples=[doc]))[0])
        doc["expected"] = '{"status":"nao_atendido"}'
        self.assertFalse(quality_key(dict(samples=[doc]))[0])
        with self.assertRaises(ValueError): quality_key(dict(samples=[]))

