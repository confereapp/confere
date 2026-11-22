import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_requirement_repetition import repeated_prompt
from confere_ai.grounding import prompt_for


class RequirementRepetitionTests(unittest.TestCase):
    def test_repetition_preserves_every_source_character_and_original_requirement(self):
        row = dict(requirement="O projeto deve incluir testes.", units=[dict(id="page:1", text="🙂 Há documentação; não há testes.")], grounding_schema=2)
        row["prompt"] = prompt_for(row["requirement"], row["units"], 2)
        original = row["prompt"]
        result = repeated_prompt(row)
        self.assertEqual(result, original + "\nRequisito a conferir: " + row["requirement"])
        self.assertEqual(row["prompt"], original)
        with self.assertRaises(ValueError): repeated_prompt(dict(row, prompt="outro texto"))
