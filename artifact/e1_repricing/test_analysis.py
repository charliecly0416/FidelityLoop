"""Self-contained checks for arithmetic, invalid inputs, and package integrity."""
import copy
import json
import shutil
from pathlib import Path
from statistics import mean
import tempfile
import unittest
import analysis
import verify


class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=analysis.read_data()
        cls.result=analysis.calculate(cls.data)

    def test_reference_and_denominators(self):
        report=verify.compare(self.result,json.loads((analysis.HOME/'reference_results.json').read_text()))
        self.assertLessEqual(report['maximum_absolute_error'],1e-12)
        self.assertEqual([self.result[k] for k in ('formal_count','comparison_count','excluded_capacity_count')],[14,12,2])
        self.assertEqual(len(self.result['results']),4)

    def test_repricing_from_individual_ledgers(self):
        for r in self.result['results']:
            for x,y in [(1,0),(1,1),(.5,2),(1,r['a_star_at_g_1']),(r['g_star_at_a_1'],1)]:
                means={}
                for arm in analysis.ARMS:
                    group=[v for v in self.data['ledger'] if v['kind']=='comparison' and v['window_id']==r['window'] and v['policy_id']==arm]
                    values=[]
                    for v in group:
                        p=r['phase']+'_cost_'
                        values.append(x*float(v[p+'gpu'])+y*float(v[p+'synthetic_api'])+float(v[p+'startup'])+float(v[p+'shutdown'])+float(v[p+'offline_miss']))
                    self.assertEqual(len(values),3);means[arm]=mean(values)
                self.assertAlmostEqual(means['H+guard']-means['B-HPA'],analysis.scenario_difference(r['coefficients_USD'],x,y),places=12)

    def test_both_sides_of_each_boundary(self):
        for r in self.result['results']:
            for axis,key,orientation in [('api','a_star_at_g_1',1),('gpu','g_star_at_a_1',-1)]:
                for direction in (-1,1):
                    value=analysis.scenario_difference(r['coefficients_USD'],**{axis:r[key]+direction*.005})
                    self.assertGreater(value*orientation*direction,0)

    def test_equal_window_weighting(self):
        rows=[r for r in self.result['results'] if r['phase']=='window']
        equal=mean(r['reference_saving_fraction'] for r in rows)
        pooled=1-sum(r['arm_means']['H+guard']['total_cost'] for r in rows)/sum(r['arm_means']['B-HPA']['total_cost'] for r in rows)
        self.assertEqual(equal,self.result['equal_window_mean_window_cost_saving_fraction'])
        self.assertGreater(abs(equal-pooled),.001)

    def test_identity_and_repetition_errors(self):
        for change in ('missing','duplicate','repeat'):
            with self.subTest(change=change):
                data=copy.deepcopy(self.data);rows=data['ledger'];target=next(v for v in rows if v['kind']=='comparison')
                if change=='missing':rows.remove(target)
                if change=='duplicate':target['run_id']=rows[0]['run_id']
                if change=='repeat':target['repeat']='2' if target['repeat']=='1' else '1'
                with self.assertRaises(ValueError):analysis.calculate(data)

    def test_inconsistent_bookkeeping(self):
        for field in ('window_cost_gpu','full_deployment_total_cost','window_penalty_cost','offline_timely'):
            with self.subTest(field=field):
                data=copy.deepcopy(self.data);target=next(v for v in data['ledger'] if v['kind']=='comparison')
                target[field]=str(float(target[field])+1)
                with self.assertRaises(ValueError):analysis.calculate(data)

    def test_request_projection_errors(self):
        for change in ('population','route','tokens','component'):
            with self.subTest(change=change):
                data=copy.deepcopy(self.data);row=next(v for v in data['ledger'] if v['kind']=='comparison' and v['policy_id']=='H+guard');detail=data['routes'][row['run_id']]['independent'];key=next(k for k,v in detail['requests'].items() if v['route']=='synthetic_api')
                if change=='population':del detail['requests'][key]
                if change=='route':detail['requests'][key]['route']='gpu0'
                if change=='tokens':data['requests'][row['window_id']][key]['input_tokens']+=1
                if change=='component':detail['phases']['window']['cost_components']['gpu']+=1
                with self.assertRaises(ValueError):analysis.calculate(data)

    def test_invalid_prices(self):
        coeff=self.result['results'][0]['coefficients_USD']
        for axis in ('gpu','api'):
            for value in (-1,float('nan'),float('inf')):
                with self.subTest(axis=axis,value=value),self.assertRaises(ValueError):analysis.scenario_difference(coeff,**{axis:value})
        data=copy.deepcopy(self.data);data['rates']['gpu_second']=-1
        with self.assertRaises(ValueError):analysis.calculate(data)

    def test_current_manifest(self):
        verify.check_manifest()

    def test_modified_or_extra_members(self):
        for change in ('modified','extra'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/'candidate';shutil.copytree(analysis.HOME,directory)
                if change=='modified':(directory/'data/ledger.csv').write_text('changed')
                if change=='extra':(directory/'unexpected.txt').write_text('unexpected')
                with self.assertRaises(ValueError):verify.check_manifest(directory)


if __name__ == '__main__':
    unittest.main()
