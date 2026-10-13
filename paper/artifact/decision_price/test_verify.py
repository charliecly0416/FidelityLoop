import unittest
from verify import choose, assess, minimizing_set, reprice, COMPONENTS


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


class PriceTests(unittest.TestCase):
    def test_reprices_only_gpu_and_api_without_mutating_input(self):
        row = {'P1plus': False, 'costs': dict(zip(COMPONENTS, [10, 2, 3, 20, 4]))}
        priced = reprice(row, 2, 0.5)
        self.assertEqual(priced['cost_total'], 20+2+3+10+4)
        self.assertFalse(priced['P1plus'])
        self.assertEqual(row['costs']['gpu'], 10)
        self.assertEqual(row['costs']['synthetic_api'], 20)

    def test_reselects_after_price_change(self):
        a = {'P1plus': True, 'costs': dict(zip(COMPONENTS, [10, 0, 0, 1, 0]))}
        b = {'P1plus': True, 'costs': dict(zip(COMPONENTS, [1, 0, 0, 10, 0]))}
        def predictions(gpu, api):
            return {('a', 'w'): reprice(a, gpu, api), ('b', 'w'): reprice(b, gpu, api)}
        self.assertEqual(choose(predictions(.5, 2), ['a', 'b'], ['w']), 'a')
        self.assertEqual(choose(predictions(2, .5), ['a', 'b'], ['w']), 'b')
        self.assertEqual(minimizing_set(predictions(1, 1), ['a', 'b'], ['w']), ['a', 'b'])

    def test_reprices_physical_optimum_before_comparison(self):
        a = {'P1plus': True, 'costs': dict(zip(COMPONENTS, [10, 0, 0, 1, 0]))}
        b = {'P1plus': True, 'costs': dict(zip(COMPONENTS, [1, 0, 0, 10, 0]))}
        physical = {('a', 'w'): [reprice(a, 2, .5)], ('b', 'w'): [reprice(b, 2, .5)]}
        out = assess('a', physical, ['a', 'b'], ['w'])
        self.assertEqual(out['best_observed_feasible_policy'], 'b')
        self.assertEqual(out['cost_regret_scenario_usd'], 13.5)

    def test_incomplete_components_rejected(self):
        with self.assertRaises(ValueError):
            reprice({'costs': {'gpu': 10}}, 1, 1)


if __name__ == '__main__':
    unittest.main()

