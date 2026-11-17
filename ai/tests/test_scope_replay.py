import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_query_contrasts import scoped_conversations
from prepare_scope_replay import merge_scopes
from probe_conversation_routing import presence_pools


class ScopeReplayTests(unittest.TestCase):
    def test_reserved_requests_are_excluded_and_training_conflicts_fail(self):
        added=scoped_conversations(1)[4:7]
        source=[dict(added[0],split='validation'),dict(added[1],id='existing',project_id='existing')]
        original=copy.deepcopy(source)
        rows,counts=merge_scopes(source,added,'fresh')
        self.assertEqual(counts,dict(reserved_collision=1,duplicate_train=1,added_train=1))
        self.assertEqual(source,original)
        self.assertEqual(rows[:len(source)],source)
        with self.assertRaises(ValueError):merge_scopes([dict(added[0],target='outro alvo')],added,'fresh')

    def test_new_namespaces_preserve_bodies_and_contrasting_answers(self):
        added=scoped_conversations(1)[4:]
        original=copy.deepcopy(added)
        rows,counts=merge_scopes([],added,'fresh')
        self.assertEqual(added,original)
        self.assertEqual(counts['added_train'],6)
        self.assertTrue(all(r['scenario']['paired_presence_group'].startswith('fresh-') for r in rows))
        pools=presence_pools(rows)
        self.assertEqual(len(pools),2)
        for pool in pools:
            self.assertEqual(len({rows[i]['scenario']['text'] for i in pool}),1)
            self.assertEqual({rows[i]['target'] for i in pool},{'Sim.','Não.','Não tenho informação suficiente para responder.'})
