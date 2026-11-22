import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_checkpoint_blend import blend_states


class CheckpointBlendTests(unittest.TestCase):
    def test_mean_preserves_sources_and_rejects_incompatible_or_invalid_weights(self):
        left = dict(weight=torch.tensor([1., 3.]), counter=torch.tensor(2))
        right = dict(weight=torch.tensor([3., 5.]), counter=torch.tensor(2))
        result = blend_states(left, right)
        self.assertTrue(torch.equal(result["weight"], torch.tensor([2., 4.])))
        self.assertTrue(torch.equal(left["weight"], torch.tensor([1., 3.])))
        self.assertEqual(result["counter"].item(), 2)
        for invalid in [{}, dict(right, weight=torch.tensor([1.])), dict(right, counter=torch.tensor(3)),
                        dict(right, weight=torch.tensor([float("nan"), 1.])),
                        dict(right, weight=right["weight"].double())]:
            with self.assertRaises(ValueError): blend_states(left, invalid)
