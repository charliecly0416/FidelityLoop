"""Export and independently check the frozen raw/derived burstiness disclosure."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def export(lock, output):
    lock, output = Path(lock), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows=[];sources={};checks=0
    for regime in ('steady','recovery','burst_offline'):
        window='locked_test_'+regime+'_v2'
        p=lock/(window+'_mapping_statistics.json')
        source=json.loads(p.read_text());sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for version in ('raw','derived'):
            for job in ('online','offline','all'):
                for width in ('1','60'):
                    s=source['statistics'][version][job][width]
                    x=s['counts_including_empty_bins'];mean=statistics.mean(x)
                    variance=sum((v-mean)**2 for v in x)
                    calculated=dict(arrival_count=sum(x),mean_count_per_bin=mean,
                        cv=statistics.pstdev(x)/mean if mean else None,
                        peak_to_mean=max(x)/mean if mean else None,
                        lag1_autocorrelation=sum((a-mean)*(b-mean) for a,b in zip(x,x[1:]))/variance if variance else None)
                    for key,value in calculated.items():
                        if (value is None)!=(s[key] is None) or (value is not None and not math.isclose(value,s[key],abs_tol=1e-9,rel_tol=1e-9)):
                            raise ValueError(f'{window}/{version}/{job}/{width}/{key}')
                        checks+=1
                    rows.append(dict(window=window,version=version,job_type=job,bin_seconds=int(width),
                                     horizon_seconds=s['horizon_seconds'],**calculated))
    with (output/'raw_derived_burstiness.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS',statistic_checks=checks,rows=len(rows),sources=sources,
                scope='frozen workload statistics, not newly generated or selected requests',
                source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                files={'raw_derived_burstiness.csv':hashlib.sha256((output/'raw_derived_burstiness.csv').read_bytes()).hexdigest()})
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.lock,a.output)
