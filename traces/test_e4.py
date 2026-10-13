"""Check E4 pairing, technical exclusions, and distinct release/scoring scopes."""
import copy
import unittest
from unittest.mock import patch
from evaluate import evaluate
from load_trace import ROOT, read_index
from verify import check_e4_inventory


class E4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = next(c for c in read_index()['campaigns'] if c['campaign'] == 'E4')

    def evaluate_e4(self, campaign):
        with patch('evaluate.read_index', return_value={'campaigns': [campaign]}):
            return evaluate(ROOT)['groups']

    def test_released_anchors_and_reused_predictions_do_not_inflate_denominator(self):
        c = self.campaign
        self.assertEqual(len(c['physical']), 26)
        self.assertEqual(len(c['predictions']), 42)
        self.assertEqual(len(c['pairs']), 78)
        for row in self.evaluate_e4(c):
            self.assertEqual(row['comparison_cells'], 24)
        self.assertEqual(len({p['prediction_id'] for p in c['pairs']}), 42)

    def test_mispaired_or_duplicated_predictions_rejected(self):
        c = copy.deepcopy(self.campaign)
        c['pairs'][0]['prediction_id'] = c['pairs'][-1]['prediction_id']
        with self.assertRaisesRegex(ValueError, 'pair index'):
            self.evaluate_e4(c)
        c = copy.deepcopy(self.campaign)
        c['pairs'].append(copy.deepcopy(c['pairs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate physical/prediction'):
            self.evaluate_e4(c)

    def test_exclusions_keep_replacement_links_and_no_claimed_root_cause(self):
        check_e4_inventory(self.campaign)
        c = copy.deepcopy(self.campaign)
        c['excluded_attempts'][0]['replacement_physical_id'] = 'missing'
        with self.assertRaises(ValueError): check_e4_inventory(c)
        c = copy.deepcopy(self.campaign)
        c['excluded_attempts'][0]['root_cause'] = 'environment'
        with self.assertRaises(ValueError): check_e4_inventory(c)

    def test_duplicate_coordinates_are_not_new_experiments(self):
        c = copy.deepcopy(self.campaign)
        c['predictions'].append(copy.deepcopy(c['predictions'][0]))
        with self.assertRaisesRegex(ValueError, 'coordinate'): check_e4_inventory(c)

if __name__ == '__main__':
    unittest.main()
