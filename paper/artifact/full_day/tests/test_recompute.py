import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recompute',ROOT/'tools/recompute.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.expected={'a':dict(job_type='online',arrival_s=0,deadline_s=60),
                       'b':dict(job_type='offline',arrival_s=86385,deadline_s=86685)}

    def test_exact_deadline_and_full_denominator(self):
        result=m.score(self.expected,{'a':60,'b':86685})
        self.assertTrue(result['P1plus'])
        result=m.score(self.expected,{'a':60})
        self.assertTrue(result['online_only']);self.assertFalse(result['P1plus'])
        self.assertEqual(result['populations']['offline']['unfinished'],1)

    def test_grace_does_not_restore_unfinished_or_cutoff(self):
        self.assertFalse(m.score(self.expected,{'a':60,'b':86701},100)['P1plus'])
        self.assertFalse(m.score(self.expected,{'a':60,'b':None},100)['P1plus'])
        self.assertTrue(m.score(self.expected,{'a':60,'b':86690},10)['P1plus'])
        self.assertFalse(m.score(self.expected,{'a':60.01,'b':86690},10)['online_only'])

    def test_invalid_completion_and_grace(self):
        for completed in ({'x':1},{'a':-1},{'a':float('nan')}):
            with self.assertRaises(ValueError):m.score(self.expected,completed)
        for grace in (-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):m.score(self.expected,{},grace)

    def timeline(self):
        # Independent hand-computed occupancy: GPU0 covers [0,10], GPU1 [5,15]:20s.
        events=[]
        for seq,kind,gpu,at in [(0,'launch_issued',0,-2),(1,'launch_issued',1,5),
                               (2,'ready',0,6),(3,'ready',1,8),
                               (4,'verified_release',0,10),(5,'verified_release',1,15)]:
            row=dict(seq=seq,kind=kind,gpu=gpu,generation=1,at_ns=at*m.NS,
                     phase='setup' if at<0 else 'window')
            if kind=='verified_release':row['release_verified']=True
            events.append(row)
        return events

    def test_independent_clipped_interval_truth(self):
        result=m.occupied(self.timeline())
        self.assertEqual(result['gpu_seconds'],20)
        self.assertEqual(result['per_gpu_seconds'],{'0':10,'1':10})
        self.assertEqual(result['startup_events'],1)
        self.assertEqual(result['startup_sample_count'],1)
        self.assertEqual(result['startup_p95'],3)

    def test_missing_release_duplicate_and_unverified_rejected(self):
        original=self.timeline()
        bad=[original[:-1], original+[dict(original[-1],seq=6)],
             original[:-1]+[dict(original[-1],release_verified=False)]]
        for events in bad:
            with self.assertRaises(ValueError):m.occupied(events)

    def test_same_gpu_generation_overlap_rejected(self):
        events=self.timeline()
        events[1]['gpu']=0;events[1]['generation']=2
        events[3]['gpu']=0;events[3]['generation']=2
        events[5]['gpu']=0;events[5]['generation']=2
        with self.assertRaises(ValueError):m.occupied(events)

    def test_abstention_ties_and_missing_pool(self):
        rows=[dict(day=day,model=model,policy=policy,P1plus=True,frozen_scenario_cost=1.)
              for day in ('D1','D2') for model in ('Share','Lookup69','Lookup30','Local')
              for policy in ('H','Hguard')]
        self.assertEqual(m.selections(rows)['Local'],['H','Hguard'])
        for r in rows:
            if r['day']=='D1':r['P1plus']=False
        self.assertEqual(m.selections(rows)['Local'],[])
        with self.assertRaises(ValueError):m.selections(rows[:-1])


if __name__=='__main__':unittest.main()
