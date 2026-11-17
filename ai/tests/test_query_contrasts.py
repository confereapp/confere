import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_query_contrasts import conversations, documents, long_documents, intent_variations, presence_pairs, comparison_pairs,scoped_conversations
from confere_ai.tokenizer import ByteBPE
from confere_ai.grounding import judge
from prepare_context_answers import answer
from prepare_lexical_binding import expected


class QueryContrastTests(unittest.TestCase):
    def test_qualifiers_vary_without_determining_presence_or_absence(self):
        targets={word:set() for word in ['de','por','para','com','em']}
        mixed=False
        for row in scoped_conversations(200):
            scene=row['scenario']
            if 'positive' not in scene:continue
            mixed=mixed or len(set(scene['prepositions']))>1
            for word in targets:
                if ' '+word+' ' in scene['queried']:targets[word].add(row['target'])
            expected=('Sim.' if scene['queried']==scene['positive'] else 'Não.' if scene['queried']==scene['negative']
                      else 'Não tenho informação suficiente para responder.')
            self.assertEqual(row['target'],expected)
        self.assertTrue(mixed)
        self.assertTrue(all(values=={'Sim.','Não.','Não tenho informação suficiente para responder.'} for values in targets.values()))

    def test_numeric_noise_does_not_supply_a_missing_deadline_and_membership_can_reverse(self):
        rows=scoped_conversations(10)
        for offset in range(0,len(rows),10):
            group=rows[offset:offset+10];numeric=group[:4]
            self.assertEqual(numeric[0]['target'],'Não tenho informação suficiente para responder.')
            self.assertEqual(numeric[1]['target'],numeric[2]['target'])
            self.assertEqual(numeric[3]['target'],'Há informações conflitantes no texto.')
            self.assertTrue(all(r['scenario']['noise']==numeric[0]['scenario']['noise'] for r in numeric))
            for trio in [group[4:7],group[7:]]:
                self.assertEqual(len({r['scenario']['paired_presence_group'] for r in trio}),1)
                self.assertEqual(len({r['scenario']['text'] for r in trio}),1)
                self.assertEqual(len({r['target'] for r in trio}),3)
            for a,b in zip(group[4:7],group[7:]):
                self.assertEqual(a['scenario']['queried'],b['scenario']['queried'])
                if a['target']!='Não tenho informação suficiente para responder.':self.assertNotEqual(a['target'],b['target'])
    def test_repetition_and_contradiction_use_the_same_values_and_requested_field(self):
        rows=comparison_pairs(100)
        for a,b,c in zip(rows[::3],rows[1::3],rows[2::3]):
            self.assertEqual(a['target'],b['target'])
            self.assertNotEqual(b['target'],c['target'])
            self.assertEqual(len({r['scenario']['comparison_group'] for r in [a,b,c]}),1)
            for row in [a,b,c]:self.assertEqual(row['target'],answer(row['scenario']['facts'],row['scenario']['field']))
    def test_conflict_values_share_the_factual_distribution_without_a_fixed_marker(self):
        rows=[r for r in conversations(1200) if r['task'].endswith('conflict') and r['scenario']['field']=='prazo']
        pairs=[r['scenario']['facts']['prazo'] for r in rows]
        self.assertGreater(len({values[-1] for values in pairs}),20)
        self.assertTrue(all(1<=v<=180 for values in pairs for v in values))
        self.assertTrue(all(len(set(values))==2 for values in pairs))
    def test_sentence_units_do_not_inherit_another_sentences_evidence(self):
        rows=documents(40,topics=['consulta de obras','cadastro de animais','edição de visitas'],sentence_units=True)
        self.assertTrue(any(len(r['units'])>2 for r in rows))
        for row in rows:
            self.assertLessEqual(len(row['units']),4)
            self.assertTrue(all(len(u['facts'])<=1 for u in row['scenario']['units']))
            self.assertEqual(json.loads(row['target']),expected(row['scenario']['units'],row['scenario']['feature']))

    def test_multiword_curriculum_preserves_feature_specific_annotations(self):
        topics=['registro de visitas','controle de acesso','catálogo de obras']
        rows=documents(20,topics=topics,forms=[('Foi entregue {}.','Não foi entregue {}.')])
        for row in rows:
            self.assertIn(row['scenario']['feature'],topics)
            self.assertEqual(json.loads(row['target']),expected(row['scenario']['units'],row['scenario']['feature']))

    def test_same_document_requires_different_answers_without_lexical_shortcuts(self):
        rows = documents(30)
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual(a["units"], b["units"])
            for row in [a, b]:
                self.assertEqual(json.loads(row["target"]), expected(row["scenario"]["units"], row["scenario"]["feature"]))
        self.assertTrue(any(json.loads(a["target"]) != json.loads(b["target"]) for a, b in zip(rows[::2], rows[1::2])))

    def test_unrelated_conflict_does_not_contaminate_other_questions(self):
        rows = conversations(24)
        rows = [r for r in rows if "scenario" in r]
        for row in rows:
            self.assertEqual(row["target"], answer(row["scenario"]["facts"], row["scenario"]["field"]))
        for offset in range(0, len(rows), 4):
            group = rows[offset:offset+4]
            self.assertTrue(all(r['scenario']['facts']==group[0]['scenario']['facts'] for r in group))
            self.assertEqual(len({r["scenario"]["field"] for r in group}), 4)
        conflict_groups = [rows[i:i+4] for i in range(0, len(rows), 4) if any(r["task"].endswith("conflict") for r in rows[i:i+4])]
        self.assertTrue(conflict_groups)
        self.assertTrue(all(sum(r["task"].endswith("conflict") for r in g) == 1 for g in conflict_groups))

    def test_facts_vary_in_count_and_questions_are_not_bound_to_newlines(self):
        rows=[r for r in conversations(200) if not r['task'].endswith('memory')]
        self.assertTrue(any(len(r['scenario']['facts'])==1 and r['task'].endswith('conflict') for r in rows))
        self.assertTrue(any('\n' not in r['prompt'] for r in rows))
        self.assertTrue(any('\nPergunta: ' in r['prompt'] for r in rows))

    def test_long_transfer_uses_actual_fragments_and_never_labels_partial_sentences(self):
        rows, skipped = long_documents(ByteBPE([]), 1)
        self.assertTrue(rows)
        self.assertGreaterEqual(skipped, 0)
        self.assertTrue(all(r["split"] == "train" and r["id"].startswith("query-long-") for r in rows))
        self.assertTrue(all(judge(r["target"], r)["exact_match"] for r in rows))
        self.assertTrue(any(r["training_profile"] == "scope_long" for r in rows))

    def test_intent_variations_keep_target_and_do_not_copy_validation_requests(self):
        source = [dict(id="train", project_id="train", split="train", task="conversation_slang_function",
                       prompt="o que você faz?", target="Sou um experimento para apoiar a conferência de projetos.", history=[], turn_index=0),
                  dict(id="val", project_id="val", split="validation", task="conversation_slang_function",
                       prompt="o q tu faz aí?", target="Sou um experimento para apoiar a conferência de projetos.", history=[], turn_index=0)]
        rows = intent_variations(source, 40)
        self.assertTrue(rows)
        self.assertTrue(all(r["prompt"] != source[1]["prompt"] for r in rows))
        self.assertTrue(all(r["split"] == "train" for r in rows))
        transformed = [r for r in rows if r["id"].startswith("intent-variation-")]
        self.assertTrue(transformed)
        self.assertTrue(all(r["target"] == source[0]["target"] and r["turn_index"] == 0 for r in transformed))

    def test_memory_corrections_use_latest_name_and_coherent_history(self):
        rows = [r for r in conversations(4) if r["task"].endswith("memory")]
        self.assertEqual(len(rows), 6)
        for a, b, c in zip(rows[::3], rows[1::3], rows[2::3]):
            self.assertNotEqual(a["target"], b["target"])
            self.assertEqual(c["target"], "Seu nome é " + b["target"].removeprefix("Prazer, "))
            self.assertEqual(c["history"], [[a["prompt"], a["target"]], [b["prompt"], b["target"]]])

    def test_resource_spelling_does_not_determine_polarity(self):
        rows = documents(300)
        near = [r for r in rows if r["scenario"]["feature"].startswith(("im", "ex", "pré", "pós"))]
        states = {e["status"] for r in near for e in json.loads(r["target"])["evidence"]}
        self.assertEqual(states, {"atendido", "nao_atendido"})

    def test_presence_questions_require_opposite_answers_for_identical_context(self):
        rows = presence_pairs(20)
        for a, b, c in zip(rows[::3], rows[1::3], rows[2::3]):
            self.assertEqual([a["target"], b["target"], c["target"]], ["Sim.", "Não.", "Não tenho informação suficiente para responder."])
            self.assertEqual(a["scenario"]["text"], b["scenario"]["text"])
            self.assertEqual(a["scenario"]["text"], c["scenario"]["text"])

    def test_near_resources_vary_beyond_two_fixed_prefixes(self):
        rows = documents(120)
        pairs = [(rows[i*2]["scenario"]["feature"], rows[i*2+1]["scenario"]["feature"]) for i in range(120) if i % 3 == 1]
        self.assertGreater(len({a[:2] for a, _ in pairs}), 5)
        self.assertTrue(all(a != b for a, b in pairs))

    def test_question_form_is_not_correlated_with_clause_order(self):
        rows = presence_pairs(2000)
        direct_positive_after_negative = [r for r in rows if r["target"] == "Sim."
            and r["scenario"]["text"].startswith("O projeto não inclui") and ", mas " in r["scenario"]["text"]
            and r["prompt"].endswith("A documentação foi incluída?")]
        self.assertTrue(direct_positive_after_negative)
        from evaluate_context_smoke import CASES
        self.assertFalse({r["prompt"] for r in rows} & {c["message"] for c in CASES})
