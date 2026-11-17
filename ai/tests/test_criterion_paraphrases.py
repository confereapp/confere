import random
import unittest
from prepare_criterion_contrasts import criterion_rows
from prepare_criterion_paraphrases import paraphrase


class CriterionParaphraseTests(unittest.TestCase):
    def test_preserves_facts_labels_and_reserved_rows(self):
        rng=random.Random(42)
        train=criterion_rows('train',5);reserved=criterion_rows('validation',5)
        changed=0
        for row in train+reserved:
            updated=paraphrase(row,rng)
            self.assertEqual(updated['units'],row['units'])
            self.assertEqual(updated['target'],row['target'])
            self.assertEqual(updated['scenario'],row['scenario'])
            if row['split']=='validation':self.assertEqual(updated,row)
            else:changed+=updated['prompt']!=row['prompt']
        self.assertGreater(changed,0)


if __name__=='__main__':unittest.main()
