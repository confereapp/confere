import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_error_replay import replay


class ErrorReplayTests(unittest.TestCase):
    def test_replays_entire_training_conversation_and_preserves_validation(self):
        rows=[dict(id='a',project_id='p',split='train',turn_index=0),
              dict(id='b',project_id='p',split='train',turn_index=1),
              dict(id='v',project_id='v',split='validation',turn_index=0)]
        result=replay(rows,{'b','v'},2)
        self.assertEqual(result[:3],rows)
        self.assertEqual(len(result),7)
        self.assertTrue(all(r['split']=='train' for r in result[3:]))
        self.assertEqual([r['turn_index'] for r in result[3:]],[0,1,0,1])
        self.assertEqual(len({r['id'] for r in result}),7)
