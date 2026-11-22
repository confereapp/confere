import copy
import json
from pathlib import Path
import unittest

from confere_ai.relation_partition import components, verify_partition, split_for, PROTOCOL
from prepare_relation_documents import paragraph_rows, assign_partitions
from confere_ai.paragraph_source import validate_provenance


def item(key, query, text, split='train', document=None):
    result = dict(id=key, requirement=query, units=[dict(id='page:1',text=text)],
                  split=split,provenance=dict(partition_protocol=PROTOCOL))
    if document: result['source_document'] = document
    return result


class RelationPartitionTests(unittest.TestCase):
    def test_cross_role_near_duplicate_and_transitive_document_links(self):
        rows = [item('a','A consulta original pergunta algo.','Coração azul sobre mesa grande e antiga.'),
                item('b','coracao azul sobre mesa grande e antiga!','Outro trecho bem diferente.'),
                item('c','Outra consulta independente.','Sem vínculo lexical algum.',document='D'),
                item('d','Outro trecho bem diferente.','Conteúdo inteiramente novo.',document='D'),
                item('e','Uma entidade desliga sensores remotos.','Uma ave cruza oceanos distantes.')]
        keys = components(rows)
        self.assertEqual(len({keys[r['id']] for r in rows[:4]}),1)
        self.assertNotEqual(keys['a'],keys['e'])
        rows[1]['split'] = 'test'
        with self.assertRaisesRegex(ValueError,'atravessa splits'): verify_partition(rows)

    def test_lexical_near_duplicates_cannot_cross_split_and_order_is_deterministic(self):
        rows=[item('a','consulta de teste','um carro vermelho cruza a ponte velha sobre rio hoje'),
              item('b','pergunta distinta','um carro vermelho cruza a ponte velha sobre rio agora')]
        self.assertEqual(components(rows)['a'],components(rows)['b'])
        first=assign_partitions(copy.deepcopy(rows));second=assign_partitions(list(reversed(copy.deepcopy(rows))))
        self.assertEqual({r['id']:r['split'] for r in first},{r['id']:r['split'] for r in second})
        self.assertEqual(split_for(first[0]['project_id']),first[0]['split'])

    def test_editorial_paragraph_keeps_literal_source_and_anchor(self):
        context='A instituição é pública e fica em uma cidade.'
        source=dict(data=[dict(title='DOC',paragraphs=[dict(context=context)])])
        annotations=dict(documents=[dict(title='DOC',paragraph_index=0,split='train',anchor='é pública',
            claims=[dict(text='A instituição é pública.',label=1,rationale='Afirmação explícita.')])])
        row=paragraph_rows(source,annotations)[0]
        validate_provenance(row)
        self.assertEqual(row['units'][0]['text'],context)
        changed=copy.deepcopy(row);changed['units'][0]['text']+=' Extra.'
        with self.assertRaises(ValueError): validate_provenance(changed)
        changed=copy.deepcopy(row);changed['provenance']['anchor_start']=0
        with self.assertRaises(ValueError): validate_provenance(changed)

    def test_frozen_annotation_documents_have_no_split_overlap(self):
        path=Path(__file__).resolve().parents[1]/'data/paragraph-criteria-v1.json'
        docs=json.loads(path.read_text())['documents'];splits={}
        for d in docs:
            self.assertEqual(splits.setdefault(d['title'],d['split']),d['split'])
            self.assertEqual({c['label'] for c in d['claims']},{0,1,2})


if __name__=='__main__': unittest.main()
