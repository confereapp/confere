import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_task_curricula import partition


class TaskCurriculaTests(unittest.TestCase):
    def test_partition_preserves_all_rows_and_validation_without_cross_task_loss(self):
        rows = [dict(id=str(i), task=task, split=split) for i, (task, split) in enumerate(
            [("grounded_atendido", "train"), ("grounded_nao_atendido", "validation"),
             ("conversation_definition", "train"), ("conversation_context_answer_fact", "validation")])]
        result = partition(rows)
        self.assertEqual(result["documents"], rows[:2])
        self.assertEqual(result["conversation"], rows[2:])
        self.assertEqual(sum(map(len, result.values())), len(rows))
        with self.assertRaises(ValueError): partition(rows[:2])
        with self.assertRaises(ValueError): partition(rows + [dict(task="public_dialogue")])
