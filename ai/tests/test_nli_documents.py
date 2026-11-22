import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from confere_ai.instruct import load_instructions
from confere_ai.nli_source import validate_provenance

spec = importlib.util.spec_from_file_location('prepare_nli_documents', Path(__file__).resolve().parents[1]/'scripts/prepare_nli_documents.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class NliDocumentsTests(unittest.TestCase):
    def row(self, label):
        return builder.annotated_row(dict(sentence_pair_id=str(label+1), premise='Uma pessoa está sorrindo.', hypothesis='A pessoa está alegre.', label=str(label)))

    def test_source_labels_preserve_unknown_as_no_evidence(self):
        targets = [json.loads(self.row(i)['target']) for i in range(3)]
        self.assertEqual([t['status'] for t in targets], ['nao_atendido','atendido','evidencia_insuficiente'])
        self.assertEqual(targets[2]['evidence'], [])
        self.assertEqual(targets[0]['evidence'], [dict(status='nao_atendido',unit='page:1')])

    def test_source_contract_rejects_wrong_labels_license_and_reserved_splits(self):
        row = self.row(1)
        validate_provenance(row)
        for field,value in [('revision','main'),('license','Apache-2.0'),('source_label',0),('train_sha256','changed')]:
            bad=copy.deepcopy(row);bad['provenance'][field]=value
            with self.assertRaises(ValueError): validate_provenance(bad)
        row['split']='validation'
        with self.assertRaises(ValueError): validate_provenance(row)

    def test_loader_accepts_tracked_training_and_preserves_grounding_checks(self):
        row=self.row(1); fixture=copy.deepcopy(row)
        fixture.update(id='local-val',project_id='local-val',split='validation',requirement='Outro requisito.')
        from confere_ai.grounding import prompt_for
        fixture['prompt']=prompt_for(fixture['requirement'],fixture['units'],2)
        fixture['provenance']=dict(kind='synthetic_authored')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [row,fixture]))
            self.assertEqual(len(load_instructions(path)[0]),2)
            row['target']=json.dumps(dict(status='atendido',evidence=[dict(status='atendido',unit='page:99')]))
            path.write_text('\n'.join(json.dumps(r) for r in [row,fixture]))
            with self.assertRaises(ValueError):load_instructions(path)

    def test_grouped_source_requires_tracked_component_and_loader_blocks_leakage(self):
        from prepare_relation_documents import assign_partitions
        first=self.row(1);second=self.row(2)
        second['requirement']='A pessoa está ouvindo música.'
        from confere_ai.grounding import prompt_for
        second['prompt']=prompt_for(second['requirement'],second['units'],2)
        rows=assign_partitions([first,second])
        for row in rows: validate_provenance(row)
        bad=copy.deepcopy(rows[0]);bad['provenance']['component_sha256']='changed'
        with self.assertRaises(ValueError): validate_provenance(bad)
        rows[0]['split']='train';rows[1]['split']='validation'
        # Mesmo que os IDs de projeto sejam falsificados, o texto compartilhado denuncia vazamento.
        rows[1]['project_id']='f'*64;rows[1]['provenance']['component_sha256']='f'*64
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'atravessa splits'): load_instructions(path)
