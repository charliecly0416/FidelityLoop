"""Semantic checks for the additional frozen campaigns and the public example."""
import copy
import unittest
from load_trace import read_index
from test_trace import fixture
from verify import check_extension_inventory
from worked_example import inspect, pair_measures


def paired_fixture():
    p = fixture()
    p.update(campaign='example', policy='H', window='w', kind='physical',
             record_id='p', reference={})
    q = copy.deepcopy(p)
    q.update(kind='prediction', record_id='q')
    return p, q


class ExtensionTests(unittest.TestCase):
    def test_pair_identity_and_contract_rejected(self):
        for field, value in [('window', 'other'), ('policy', 'other'),
                             ('campaign', 'other'), ('horizon_s', 11)]:
            p, q = paired_fixture()
            q[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pair_measures(p, q)
        p, q = paired_fixture()
        q['prices']['gpu_second'] = 3
        with self.assertRaises(ValueError):
            pair_measures(p, q)

    def test_budget_mismatch_rejected(self):
        p, q = paired_fixture()
        q['requests'][0]['max_output_tokens'] += 1
        with self.assertRaises(ValueError):
            pair_measures(p, q)

    def test_missing_completion_is_false_accept_not_dropped(self):
        p, q = paired_fixture()
        p['requests'][1].update(status='unfinished', completed_at_s=None)
        a, b = pair_measures(p, q)
        self.assertFalse(a['P1plus'])
        self.assertTrue(b['P1plus'])
        self.assertEqual(a['populations']['offline']['arrivals'], 1)

    def test_no_failure_cohort_has_undefined_false_accept_rate(self):
        rows = inspect('E5')['summaries']
        self.assertTrue(all(r['false_accept_rate'] is None for r in rows))
        self.assertTrue(all(r['compared_runs'] == 24 and r['unique_predictions'] == 12 for r in rows))

    def test_e6_anchors_excluded_and_failed_repeats_retained(self):
        result = inspect('E6')
        rows = {r['replay_model']: r for r in result['summaries']}
        self.assertEqual(rows['S14']['false_accepts'], 4)
        self.assertEqual(rows['S14']['compared_runs'], 8)
        self.assertEqual(rows['S14']['unique_predictions'], 4)
        self.assertEqual(rows['L14']['agreement'], 8)
        self.assertEqual(len(result['mismatches']), 4)

    def test_posthoc_model_cannot_enter_frozen_population(self):
        campaign = copy.deepcopy(next(c for c in read_index()['campaigns'] if c['campaign'] == 'E6'))
        campaign['predictions'][0]['replay_model'] = 'T30-14'
        with self.assertRaises(ValueError):
            check_extension_inventory(campaign)


if __name__ == '__main__':
    unittest.main()
