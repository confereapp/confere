import unittest
from unittest.mock import patch
from confere_ai.instruct import validation_losses


class ObjectiveSelectionTests(unittest.TestCase):
    def test_long_documents_do_not_dominate_balanced_checkpoint_selection(self):
        examples = [([1, 2], [-100, 1]), ([1] * 100, [1] * 100)]
        rows = [dict(task="conversation_definition"), dict(task="grounded_atendido")]
        with patch("confere_ai.instruct.evaluate", side_effect=[.8, .2]):
            result = validation_losses(None, examples, rows, "cpu", True)
        self.assertAlmostEqual(result["selection_loss"], .5)
        self.assertAlmostEqual(result["validation_loss"], 20.8 / 101)
        self.assertEqual(result["conversation_validation_loss"], .8)
        with patch("confere_ai.instruct.evaluate", return_value=.3) as evaluate:
            self.assertEqual(validation_losses(None, examples, rows, "cpu"), {"validation_loss": .3})
            evaluate.assert_called_once_with(None, examples, "cpu")
        for invalid in [[dict(task="conversation_definition")] * 2,
                        [dict(task="human_dialogue"), rows[1]]]:
            with self.assertRaises(ValueError): validation_losses(None, examples, invalid, "cpu", True)
