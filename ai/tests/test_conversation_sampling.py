import unittest
from collections import Counter
import torch
from confere_ai.instruct import conversation_indices


class ConversationSamplingTests(unittest.TestCase):
    def test_large_family_does_not_dominate_small_legacy_families(self):
        pools = dict(legacy=dict(a=[0], b=[1]), context=dict(c=list(range(2, 1002))))
        generator = torch.Generator().manual_seed(42); counts = Counter()
        for _ in range(4000):
            indices = conversation_indices(pools, 4, generator)
            family = "a" if indices[0] == 0 else "b" if indices[0] == 1 else "c"
            self.assertTrue(all((i == 0 if family == "a" else i == 1 if family == "b" else i >= 2) for i in indices))
            counts[family] += 1
        self.assertTrue(1800 < counts["c"] < 2200)
        self.assertTrue(850 < counts["a"] < 1150)
        self.assertTrue(850 < counts["b"] < 1150)
