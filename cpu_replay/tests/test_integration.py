"""Meaningful behavior checks on the real native Qwen control interface."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from controlled_vidur import ControlledVidur, make_policy, native_reference
from native_qwen import check_domain
from run_precheck import execute, rows, dump

def read(case,name): return json.loads((ROOT/'runtime'/case/name).read_text())
def lines(case,name): return [json.loads(x) for x in (ROOT/'runtime'/case/name).read_text().splitlines()]

class IntegrationTests(unittest.TestCase):
    def test_input_rejection_precedes_native(self):
        row=rows([(0,1)])[0]
        with patch('controlled_vidur.make_simulator') as native:
            for key,value in [('input_tokens',2049),('input_tokens',True),('max_output_tokens',257),
                              ('arrival_s',0.1),('request_id',''),('future_finish',1)]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    ControlledVidur([dict(row,**{key:value})],ROOT/'runtime/invalid')
            for opts in [dict(horizon=2101),dict(initial='ready'),dict(policy='typo'),dict(startup=float('nan'))]:
                with self.subTest(opts=opts),self.assertRaises(ValueError):
                    ControlledVidur([row],ROOT/'runtime/invalid',**opts)
            native.assert_not_called()

    def test_operator_domain_rejection(self):
        q=dict(compute_tokens=8192,kv_save_tokens=8192,aggregate_prefill_tokens=4096,
               aggregate_prefill_kv=0,decode_batch=4,decode_kv=2304)
        check_domain(q)
        for key in q:
            with self.subTest(key=key),self.assertRaises(ValueError):check_domain(dict(q,**{key:q[key]+1}))

    def test_native_equivalence(self):
        for case in ['static_equivalence','lifecycle_fixture']:
            self.assertEqual(read(case,'summary.json')['conditional_admission_max_error'],0)

    def test_restart_and_no_preemption(self):
        states=[e for e in lines('lifecycle_fixture','events.jsonl') if e['kind']=='state']
        self.assertTrue({'off','starting','active','draining','stopping'} <= {e['state'] for e in states})
        self.assertGreaterEqual(max(e['generation'] for e in states),1)
        self.assertTrue(all(r['num_restarts']==0 for r in read('lifecycle_fixture','native_requests.json').values()))

    def test_api_and_cutoff(self):
        r=read('api_fixture','records.json')
        api=[x for x in r.values() if x['route']=='synthetic_api']
        self.assertEqual(len(api),16);self.assertEqual(sum(x['completed_at'] is not None for x in api),8)
        self.assertTrue(all(x['job_type']=='online' for x in api))
        summary=read('cutoff_fixture','summary.json')
        self.assertEqual(summary['metrics']['completed'],0)
        self.assertEqual(summary['occupied_seconds'],2)
        self.assertGreater(summary['native_pending_events'],0)

    def test_frozen_guard_separate_from_native(self):
        cases=json.loads((ROOT/'inputs/cases.json').read_text())['cases']
        for case in cases:
            s=case['spec'];p=make_policy(s['policy'],s['arm'],s['guard_model'],s['guard_safety'])
            self.assertEqual(p.model['kind'],'n4c_progress_lookup')
            self.assertEqual(p.safety,s['guard_safety'])
            self.assertEqual(p.state['target'],2)
            self.assertEqual(p.scale,s['arm']['scale']);self.assertEqual(p.reservation,s['arm']['reservation'])

    def test_mixed_batch_native_reference(self):
        s=read('mixed_fixture','summary.json')
        self.assertGreater(s['domain_audit']['mixed_prefill_decode_batches'],0)
        self.assertEqual(s['conditional_admission_max_error'],0)

    def test_future_prefix_invariance(self):
        self.assertEqual([d for d in lines('prefix_base','decisions.jsonl') if d['time']<100],
                         [d for d in lines('prefix_future','decisions.jsonl') if d['time']<100])

    def test_all_frozen_service_predictions_unchanged(self):
        source=ROOT/'inputs'
        admissions=json.loads((source/'service_admissions.json').read_text())['groups']
        predictions=json.loads((source/'service_predictions.json').read_text())['groups']
        errors=[]
        for group,pred in zip(admissions,predictions):
            self.assertEqual(group['group_id'],pred['group_id'])
            rr=[dict(request_id=r['request_id'],arrival_s=r['arrival_s'],input_tokens=r['input_tokens'],
                     max_output_tokens=r['output_tokens']) for r in group['requests']]
            aa=[dict(time=r['arrival_s'],device_id=0,request_id=r['request_id']) for r in rr]
            with tempfile.TemporaryDirectory(prefix='qwen_service_regression_') as temp:
                native=native_reference(rr,Path(temp),aa)
            for r in pred['requests']:
                err=abs(native[r['request_id']]['completed_at']-r['predicted_completion_s'])
                self.assertLess(err,1e-12);errors.append(err)
        dump(ROOT/'analysis/SERVICE_REGRESSION.json',dict(status='PASS',groups=len(admissions),requests=len(errors),max_error_seconds=max(errors)))

    def test_all_historical_policy_decisions_recompute(self):
        cases=json.loads((ROOT/'inputs/cases.json').read_text())['cases']
        for case in cases:
            s=case['spec'];p=make_policy(s['policy'],s['arm'],s['guard_model'],s['guard_safety'])
            for d in lines(case['case_id'],'decisions.jsonl'):
                self.assertEqual(d['decision'],p.decide(d['observation']))

if __name__=='__main__':unittest.main()
