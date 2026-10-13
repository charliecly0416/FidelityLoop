"""Recompute the reported sensitivity statistics from reduced comparison rows.
This does not rerun the simulator or reconstruct physical costs from raw events.
"""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    summary=json.loads((ROOT/'RESULTS.json').read_text())
    with (ROOT/'PAIRED_RESULTS.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==19*3*24
    for point in summary['points']:
        for model,expected in point['models'].items():
            selected=[r for r in rows if float(r['startup_seconds'])==point['startup_seconds'] and r['replay_model']==model]
            assert len(selected)==24 and len({r['physical_record_id'] for r in selected})==24
            assert len({(r['policy'],r['window']) for r in selected})==12
            assert all(r['predicted_pass'] in ('True','False') and r['physical_pass'] in ('True','False') for r in selected)
            agreement=sum(r['predicted_pass']==r['physical_pass'] for r in selected)
            fa=sum(r['predicted_pass']=='True' and r['physical_pass']=='False' for r in selected)
            fr=sum(r['predicted_pass']=='False' and r['physical_pass']=='True' for r in selected)
            mae=statistics.mean(abs(float(r['predicted_cost'])-float(r['physical_cost'])) for r in selected)
            assert (agreement,fa,fr)==(expected['agreement'],expected['false_accepts'],expected['false_rejects'])
            assert math.isclose(mae,expected['cost_mae'],rel_tol=1e-12,abs_tol=1e-12)
            mismatch=sorted({r['policy']+'/'+r['window'] for r in selected if r['physical_pass']!=r['predicted_pass']})
            assert mismatch==expected['mismatched_arms']
    print('PASS_STARTUP_SENSITIVITY: 18 sweep points + calibration reference; 24 physical comparisons per model/point')

if __name__=='__main__':main()
