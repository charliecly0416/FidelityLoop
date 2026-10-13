import unittest
from verify import boolean, reduce_pairs, METRICS

class MetricSemantics(unittest.TestCase):
    def row(self,p,t):return dict(predicted_P1plus=str(p),physical_P1plus=str(t),**{k:'1' for k in METRICS})
    def test_conditioned_false_error_rates(self):
        r=reduce_pairs([self.row(True,False),self.row(False,True),self.row(True,True)])
        self.assertEqual((r['false_accept_rate'],r['false_reject_rate']),(1,.5))
        self.assertEqual(r['agreement'],1)
    def test_empty_negative_population(self):
        self.assertIsNone(reduce_pairs([self.row(True,True)])['false_accept_rate'])
    def test_empty_positive_population(self):
        self.assertIsNone(reduce_pairs([self.row(False,False)])['false_reject_rate'])
    def test_empty_population_rejected(self):
        with self.assertRaises(ValueError):reduce_pairs([])
    def test_truth_not_string_truthiness(self):
        self.assertFalse(boolean('False'))
        for value in ('false','0','',True):
            with self.assertRaises(ValueError):boolean(value)

if __name__=='__main__':unittest.main()
