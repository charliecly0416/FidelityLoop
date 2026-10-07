"""Independent arithmetic verifier for a reduced, fixed-event E1 dataset."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

HOME = Path(__file__).resolve().parent
SCOPES = ('setup', 'window', 'cleanup', 'full_deployment')
CATEGORIES = ('gpu', 'synthetic_api', 'startup', 'shutdown', 'offline_miss')
ARMS = ('B-HPA', 'H+guard')
WINDOWS = ('steady', 'recovery')
METRICS = ('occupied_seconds', 'active_seconds', 'startup_events', 'shutdown_events', 'api_accepted', 'operating_cost', 'penalty_cost', 'total_cost') + tuple('cost_' + c for c in CATEGORIES)


def insist(ok, explanation):
    if not ok:
        raise ValueError(explanation)


def agree(a, b, explanation):
    insist(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), explanation)


def read_data(directory=HOME):
    with (directory / 'data/ledger.csv').open(newline='', encoding='utf-8') as stream:
        ledger = list(csv.DictReader(stream))
    loaded = {'ledger': ledger}
    for key, filename in [('rates','prices'), ('routes','runs'), ('requests','requests')]:
        loaded[key] = json.loads((directory / ('data/' + filename + '.json')).read_text())
    for window, records in loaded['requests'].items():
        insist(len(records) == len({v['request_id'] for v in records}), 'request identifiers must be unique')
        for value in records:
            insist(value['job_type'] in ('online','offline'), 'unsupported request population')
            insist(all(type(value[k]) is int and value[k] >= 0 for k in ('input_tokens','max_output_tokens')), 'token budgets must be nonnegative integers')
        loaded['requests'][window] = {v['request_id']:v for v in records}
    return loaded


def audit(data):
    rows, rates, routes, requests = (data[k] for k in ('ledger','rates','routes','requests'))
    insist(set(requests) == set(WINDOWS), 'two specified windows are required')
    insist(set(rates) == {'gpu_second','api_input_token','api_output_token','startup_event','shutdown_event','offline_deadline_miss'}, 'unexpected rate definition')
    insist(all(isinstance(v,(int,float)) and math.isfinite(v) and v >= 0 for v in rates.values()), 'invalid base price')
    insist(len(rows) == 14 and len({r['run_id'] for r in rows}) == 14, 'expected fourteen distinct ledgers')
    compare = [r for r in rows if r['kind'] == 'comparison']
    gates = [r for r in rows if r['kind'] == 'capacity']
    insist(len(compare) == 12 and len(gates) == 2, 'incorrect comparison/capacity partition')
    insist({(r['window_id'],r['policy_id']) for r in compare} == {(w,a) for w in WINDOWS for a in ARMS}, 'unexpected comparison cell')
    insist({(r['window_id'],r['policy_id']) for r in gates} == {(w,'all2') for w in WINDOWS}, 'unexpected capacity gate')
    insist(set(routes) == {r['run_id'] for r in compare}, 'routed-record identities differ from comparison ledgers')
    for w in WINDOWS:
        for a in ARMS:
            repetitions = sorted(int(r['repeat']) for r in compare if (r['window_id'],r['policy_id']) == (w,a))
            insist(repetitions == [1,2,3], 'each comparison cell requires three distinct repetitions')
    for row in rows:
        insist(all(row[f] == 'True' for f in ('technical_valid','accounting_complete','strict_P1','P1_plus')), 'an input acceptance flag is not set')
        for kind in ('online','offline'):
            expected = sum(v['job_type'] == kind for v in requests[row['window_id']].values())
            insist(all(int(row[kind+'_'+x]) == expected for x in ('arrivals','completed','timely')), 'recorded population denominator is inconsistent')
            insist(all(int(row[kind+'_'+x]) == 0 for x in ('late_completed','unfinished','failed')), 'the reduced input contains an unsuccessful request')
        number = lambda scope, field: float(row[scope+'_'+field])
        for scope in SCOPES:
            parts = {c:number(scope,'cost_'+c) for c in CATEGORIES}
            insist(all(math.isfinite(v) and v >= 0 for v in parts.values()), 'invalid phase component')
            insist(parts['offline_miss'] == number(scope,'penalty_cost') == 0, 'nonzero deadline penalty')
            for cost,quantity,rate in [('gpu','occupied_seconds','gpu_second'),('startup','startup_events','startup_event'),('shutdown','shutdown_events','shutdown_event')]:
                agree(parts[cost], number(scope,quantity)*rates[rate], 'phase charge does not match quantity and base rate')
            agree(sum(parts.values()),number(scope,'total_cost'),'phase components do not sum to the total')
            agree(number(scope,'operating_cost')+number(scope,'penalty_cost'),number(scope,'total_cost'),'operating and penalty totals differ')
        for field in METRICS:
            agree(sum(number(s,field) for s in SCOPES[:3]),number('full_deployment',field),'phase partition does not add up')
        if row['kind'] == 'capacity':
            continue
        detail = routes[row['run_id']]['independent']
        population = requests[row['window_id']]
        insist(set(detail['requests']) == set(population), 'routed request population differs from input budgets')
        insist(all(v['route'] in ('gpu0','gpu1','synthetic_api') for v in detail['requests'].values()), 'unsupported fixed route')
        for scope in SCOPES:
            for category in CATEGORIES:
                agree(detail['phases'][scope]['cost_components'][category],number(scope,'cost_'+category),'CSV and reduced component record disagree')
        charged = [key for key,item in detail['requests'].items() if item['route'] == 'synthetic_api']
        rebuilt = sum(population[key]['input_tokens']*rates['api_input_token'] + population[key]['max_output_tokens']*rates['api_output_token'] for key in charged)
        agree(rebuilt,number('window','cost_synthetic_api'),'routed token budgets do not reconstruct the API charge')
        insist(len(charged) == int(row['window_api_accepted']), 'API acceptance count differs from routes')
        insist(number('setup','cost_synthetic_api') == number('cleanup','cost_synthetic_api') == 0, 'API charge outside the comparison window')
        online_api = sum(population[key]['job_type'] == 'online' for key in charged)
        agree(online_api/int(row['online_arrivals']),float(row['api_share_online_arrivals']),'online API share differs from recorded routes')
    return compare


def scenario_difference(coefficients, gpu=1.0, api=1.0):
    insist(all(math.isfinite(v) and v >= 0 for v in (gpu,api)), 'multipliers must be finite and nonnegative')
    return coefficients['delta_gpu_cost']*gpu + coefficients['delta_api_cost']*api + coefficients['delta_lifecycle_cost']


def calculate(data=None):
    data = read_data() if data is None else data
    comparisons = audit(data)
    outputs = []
    reported = ('occupied_seconds','cost_gpu','cost_synthetic_api','cost_startup','cost_shutdown','total_cost','startup_events','shutdown_events','api_accepted')
    for window in WINDOWS:
        cells = {a:[r for r in comparisons if r['window_id']==window and r['policy_id']==a] for a in ARMS}
        for scope in ('window','full_deployment'):
            averages = {a:{f:mean(float(r[scope+'_'+f]) for r in cell) for f in reported} for a,cell in cells.items()}
            differences = {f:averages[ARMS[1]][f]-averages[ARMS[0]][f] for f in reported}
            cost = {'delta_gpu_cost':differences['cost_gpu'], 'delta_api_cost':differences['cost_synthetic_api'], 'delta_lifecycle_cost':differences['cost_startup']+differences['cost_shutdown']}
            g,a,e = (cost[k] for k in ('delta_gpu_cost','delta_api_cost','delta_lifecycle_cost'))
            insist(g < 0 < a, 'expected GPU/API trade-off is absent')
            api_root, gpu_root = -(g+e)/a, -(a+e)/g
            agree(scenario_difference(cost),differences['total_cost'],'reference-price difference is not reconstructed')
            agree(scenario_difference(cost,api=api_root),0.0,'API boundary is not a zero')
            agree(scenario_difference(cost,gpu=gpu_root),0.0,'GPU boundary is not a zero')
            side = {'api_below':scenario_difference(cost,api=api_root-0.0001), 'api_above':scenario_difference(cost,api=api_root+0.0001), 'gpu_below':scenario_difference(cost,gpu=gpu_root-0.0001), 'gpu_above':scenario_difference(cost,gpu=gpu_root+0.0001)}
            insist(side['api_below']<0<side['api_above'] and side['gpu_above']<0<side['gpu_below'],'boundary-side ordering is incorrect')
            outputs.append({'window':window,'phase':scope,'n_per_arm':3,'arm_means':averages,'guard_minus_B_HPA':differences,'coefficients_USD':cost,'a_star_at_g_1':api_root,'g_star_at_a_1':gpu_root,'reference_saving_fraction':1-averages[ARMS[1]]['total_cost']/averages[ARMS[0]]['total_cost'],'root_side_delta_cost_USD':side,'online_api_share_means':{a:mean(float(r['api_share_online_arrivals']) for r in cell) for a,cell in cells.items()},'local_occupation_reduction_fraction':1-averages[ARMS[1]]['occupied_seconds']/averages[ARMS[0]]['occupied_seconds']})
    window_rows = [r for r in outputs if r['phase']=='window']
    return {'schema':'e1-reduced-fixed-event-repricing-v1','formal_count':len(data['ledger']),'comparison_count':len(comparisons),'excluded_capacity_count':len(data['ledger'])-len(comparisons),'formula':'Delta C(x,y) = x*Delta C_G + y*Delta C_A + Delta C_E; deltas are guard minus B-HPA arm means','coefficient_units':'scenario USD at registered base prices; occupied_seconds is local GPU seconds','aggregation':'three-run arm means separately within each window; equal-weight window summary, not pooled run costs','results':outputs,'equal_window_mean_window_cost_saving_fraction':mean(r['reference_saving_fraction'] for r in window_rows),'equal_window_mean_window_local_occupation_reduction_fraction':mean(r['local_occupation_reduction_fraction'] for r in window_rows)}


if __name__ == '__main__':
    print(json.dumps(calculate(), indent=2, allow_nan=False))
