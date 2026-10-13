import unittest
from verify import choose, assess, minimizing_set


class SelectionTests(unittest.TestCase):
    def test_cheaper_but_predicted_failure_is_ineligible(self):
        predictions = {('a', 'w'): {'P1plus': False, 'cost_total': 1},
                       ('b', 'w'): {'P1plus': True, 'cost_total': 2}}
        self.assertEqual(choose(predictions, ['a', 'b'], ['w']), 'b')

    def test_requires_every_window_and_complete_grid(self):
        with self.assertRaises(ValueError):
            choose({('a', 'w'): {'P1plus': True, 'cost_total': 1}}, ['a'], ['w', 'x'])
        self.assertIsNone(choose({('a', 'w'): {'P1plus': False, 'cost_total': 1}}, ['a'], ['w']))

    def test_actual_failure_has_no_feasible_cost_regret(self):
        physical = {('a', 'w'): [{'P1plus': True, 'cost_total': 1}, {'P1plus': False, 'cost_total': 1}],
                    ('b', 'w'): [{'P1plus': True, 'cost_total': 2}]}
        out = assess('a', physical, ['a', 'b'], ['w'])
        self.assertFalse(out['selected_all_repeats_feasible'])
        self.assertIsNone(out['cost_regret_scenario_usd'])

    def test_windows_equal_weight_despite_repeat_counts(self):
        physical = {('a', 'w'): [{'P1plus': True, 'cost_total': 2}] * 3,
                    ('a', 'x'): [{'P1plus': True, 'cost_total': 4}]}
        self.assertEqual(assess('a', physical, ['a'], ['w', 'x'])['selected_actual_cost'], 3)

    def test_equal_prediction_minimizers_are_retained(self):
        predictions = {('a', 'w'): {'P1plus': True, 'cost_total': 1},
                       ('b', 'w'): {'P1plus': True, 'cost_total': 1}}
        self.assertEqual(choose(predictions, ['b', 'a'], ['w']), 'a')
        self.assertEqual(minimizing_set(predictions, ['b', 'a'], ['w']), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
