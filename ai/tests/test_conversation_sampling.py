import unittest
from collections import Counter
import torch
from confere_ai.instruct import conversation_indices


class ConversationSamplingTests(unittest.TestCase):
    def test_large_family_does_not_dominate_small_legacy_families(self):
        pools = dict(legacy=dict(a=[0], b=[1]), context=dict(c=list(range(2, 1002))))
        generator = torch.Generator().manual_seed(42); counts = Counter()
        mixed = 0
        for _ in range(1000):
            indices = conversation_indices(pools, 4, generator)
            mixed += len(set("a" if i == 0 else "b" if i == 1 else "c" for i in indices)) > 1
            for i in indices:
                counts["a" if i == 0 else "b" if i == 1 else "c"] += 1
        self.assertTrue(1800 < counts["c"] < 2200)
        self.assertTrue(850 < counts["a"] < 1150)
        self.assertTrue(850 < counts["b"] < 1150)
        self.assertGreater(mixed, 800)
        self.assertEqual(conversation_indices(pools, 16, torch.Generator().manual_seed(7)),
                         conversation_indices(pools, 16, torch.Generator().manual_seed(7)))
