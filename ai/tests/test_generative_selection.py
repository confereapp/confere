import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
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

    def test_training_keeps_better_generated_answers_even_when_validation_loss_rises(self):
        from confere_ai.instruct import run
        from confere_ai.model import GPT, GPTConfig
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = GPTConfig(vocab_size=260, block_size=128, width=8, layers=1, heads=1)
            model = GPT(config)
            torch.save(dict(config=asdict(config), model=model.state_dict(), merges=[], step=0,
                            settings=dict(objective="causal_lm")), root / "source.pt")
            rows = [dict(id=split, project_id=split, split=split, prompt=prompt, target="Ok.", history=[],
                         turn_index=0, task="conversation_guidance", response_tokens=64,
                         provenance=dict(kind="synthetic_authored")) for split, prompt in [("train", "oi"), ("validation", "olá")]]
            (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = SimpleNamespace(run_dir=root / "run", data=root / "data.jsonl", init_from=root / "source.pt",
                                   steps=3, batch_size=1, eval_every=1, learning_rate=.001, device="cpu",
                                   evaluation_context=128, response_tokens=64, generative_selection=True, selection_patience=1)
            before = dict(correct=0, total=1, samples=[dict(id="validation", task="conversation_guidance", exact_match=False)])
            after = dict(correct=1, total=1, samples=[dict(id="validation", task="conversation_guidance", exact_match=True,
                                                          prompt="olá", actual="Ok.", expected="Ok.")])
            with patch("confere_ai.instruct.score", side_effect=[before, after, after]) as scoring, \
                 patch("confere_ai.instruct.evaluate", side_effect=[1., .1, .5, 10., .4, 20.]), \
                 patch("confere_ai.evaluation_cache.protocol_fingerprint", return_value="test-protocol"), \
                 patch("confere_ai.instruct.create_report", return_value=root / "unused.pdf"):
                run(args)
            result = json.loads((args.run_dir / "instruction-evaluation.json").read_text())
            self.assertEqual(result["selected_step"], 1)
            self.assertEqual(result["selected"], after)
            self.assertEqual(scoring.call_count, 3)
            self.assertTrue((args.run_dir / "generation-1.json").exists())
            summary = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(summary["final_step"], 2)
            self.assertEqual(summary["stop_reason"], "selection_plateau")

