"""Reprice frozen events and recompute all retrospective deployment choices."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parent

def choose(predictions, policies, windows):
    expected = {(p, w) for p in policies for w in windows}
    if set(predictions) != expected:
        raise ValueError('incomplete policy/window prediction grid')
    eligible = [p for p in policies if all(predictions[p, w]['P1plus'] for w in windows)]
    ranked = sorted(eligible, key=lambda p: (mean(predictions[p, w]['cost_total'] for w in windows), p))
    return ranked[0] if ranked else None


def minimizing_set(predictions, policies, windows):
    selected = choose(predictions, policies, windows)
    if selected is None:
        return []
    cost = lambda p: mean(predictions[p, w]['cost_total'] for w in windows)
    return sorted(p for p in policies if all(predictions[p, w]['P1plus'] for w in windows)
                  and cost(p) == cost(selected))


def assess(selected, physical, policies, windows):
    for p in policies:
        for w in windows:
            if not physical.get((p, w)):
                raise ValueError('missing physical comparison')
    def feasible(p):
        return all(x['P1plus'] for w in windows for x in physical[p, w])
    def cost(p):
        return mean(mean(x['cost_total'] for x in physical[p, w]) for w in windows)
    good = [p for p in policies if feasible(p)]
    best = min(good, key=lambda p: (cost(p), p)) if good else None
    out = dict(selected_policy=selected, best_observed_feasible_policy=best,
               selected_all_repeats_feasible=None, cost_regret_scenario_usd=None)
    if selected is None:
        return out
    runs = [x for w in windows for x in physical[selected, w]]
    ok = feasible(selected)
    out.update(selected_all_repeats_feasible=ok, physical_passes=sum(x['P1plus'] for x in runs),
               physical_runs=len(runs), selected_actual_cost=cost(selected))
    if ok and best is not None:
        out.update(best_observed_feasible_cost=cost(best), cost_regret_scenario_usd=cost(selected)-cost(best))
    return out


COMPONENTS = ('gpu', 'startup', 'shutdown', 'synthetic_api', 'offline_miss')


def reprice(row, gpu_multiplier, api_multiplier):
    """Reprice fixed events; startup/shutdown/miss charges remain unchanged."""
    if gpu_multiplier <= 0 or api_multiplier <= 0:
        raise ValueError('positive multipliers required')
    if set(row['costs']) != set(COMPONENTS):
        raise ValueError('incomplete cost components')
    if any(v < 0 for v in row['costs'].values()):
        raise ValueError('negative cost component')
    scaled = dict(row['costs'])
    scaled['gpu'] *= gpu_multiplier
    scaled['synthetic_api'] *= api_multiplier
    return dict(row, cost_total=sum(scaled[k] for k in COMPONENTS))


def decisions(inputs, gpu_multiplier, api_multiplier):
    output = []
    for campaign in ('E2', 'E3', 'E4', 'E5', 'E6'):
        phy, pred = defaultdict(list), defaultdict(dict)
        seen = set()
        for original in inputs:
            if original['campaign'] != campaign:
                continue
            if original['record_id'] in seen:
                raise ValueError('duplicate record')
            seen.add(original['record_id'])
            row = reprice(original, gpu_multiplier, api_multiplier)
            key = row['policy'], row['window']
            if row['kind'] == 'physical':
                phy[key].append(row)
            elif row['kind'] == 'predictions':
                if key in pred[row['replay_model']]:
                    raise ValueError('duplicate unique prediction')
                pred[row['replay_model']][key] = row
            else:
                raise ValueError('unknown record kind')
        policies = sorted({p for p, w in phy})
        windows = sorted({w for p, w in phy})
        for model, predictions in sorted(pred.items()):
            selected = choose(predictions, policies, windows)
            ties = minimizing_set(predictions, policies, windows)
            output.append(dict(campaign=campaign, model=model, candidates=len(policies), windows=len(windows),
                predicted_cost_minimizers=ties, tie_at_released_data_precision=len(ties)>1,
                minimizing_set_all_repeats_feasible={p: all(x['P1plus'] for w in windows for x in phy[p,w]) for p in ties},
                **assess(selected, phy, policies, windows)))
    return output


def run(inputs):
    return [dict(gpu_multiplier=g, api_multiplier=a, **row)
            for g in (0.5, 1, 2) for a in (0.5, 1, 2)
            for row in decisions(inputs, g, a)]


def summarize(rows):
    out = []
    for campaign, model in sorted({(r['campaign'], r['model']) for r in rows}):
        group = [r for r in rows if (r['campaign'], r['model']) == (campaign, model)]
        gaps = [100*r['cost_regret_scenario_usd']/r['best_observed_feasible_cost']
                for r in group if r['cost_regret_scenario_usd'] is not None]
        out.append(dict(campaign=campaign, model=model, price_cells=len(group),
            selected_policies=sorted({r['selected_policy'] for r in group}),
            feasible_selected_cells=sum(r['selected_all_repeats_feasible'] is True for r in group),
            failed_selected_cells=sum(r['selected_all_repeats_feasible'] is False for r in group),
            no_selection_cells=sum(r['selected_policy'] is None for r in group),
            tied_minimizer_cells=sum(r['tie_at_released_data_precision'] for r in group),
            feasible_selected_gap_pct_min=min(gaps) if gaps else None,
            feasible_selected_gap_pct_max=max(gaps) if gaps else None))
    return out


def verify():
    inputs = json.loads((ROOT/'INPUTS.json').read_text())
    for row in inputs['rows']:
        assert reprice(row, 1, 1)['cost_total'] == row['cost_total']
    assert decisions(inputs['rows'], 1, 1) == json.loads((ROOT/'NOMINAL_BASELINE.json').read_text())['rows']
    rows = run(inputs['rows'])
    expected = json.loads((ROOT/'RESULTS.json').read_text())
    assert rows == expected['rows']
    assert summarize(rows) == expected['summary']
    assert len(rows) == 108
    print(json.dumps({'status': 'PASS_FIXED_EVENT_PRICE_SENSITIVITY',
                      'source_rows': len(inputs['rows']), 'price_cells': 9,
                      'decisions': len(rows), 'nominal_decisions_restored': 12}))


if __name__ == '__main__':
    verify()

