"""Recompute retrospective deployment choices from reduced public-corpus metrics."""
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


def run():
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())['rows']
    output = []
    for campaign in ('E2', 'E3', 'E4', 'E5', 'E6'):
        phy, pred = defaultdict(list), defaultdict(dict)
        seen = set()
        for row in inputs:
            if row['campaign'] != campaign:
                continue
            if row['record_id'] in seen:
                raise ValueError('duplicate record')
            seen.add(row['record_id'])
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


def table_rows(rows):
    labels = {'ppo_C30_raw':'C30-2', 'ppo_C_r2':'C-2', 'ppo_C_s101':'C-1',
              'C30_s105_e0065':'N1', 'C30_s105_e0035':'N2',
              'hysteresis_u8_d0_c60':'H', 'hysteresis_u8_d0_c60__guard':'H+guard'}
    lines=[]
    for row in rows:
        model = r'C$_{\rm sg}$' if row['model']=='C_sg' else row['model']
        choice = labels[row['selected_policy']]
        if row['tie_at_released_data_precision']:
            choice += r'$^{\dagger}$'
        gap = '--' if row['cost_regret_scenario_usd'] is None else f"{100*row['cost_regret_scenario_usd']/row['best_observed_feasible_cost']:.2f}"
        lines.append(f"{row['campaign']} & {model} & {choice} & {row['physical_passes']}/{row['physical_runs']} & {row['selected_actual_cost']:.4f} & {gap} " + r"\\")
    return '\n'.join(lines)+r'\bottomrule'+'\n'


if __name__ == '__main__':
    rows = run()
    assert rows == json.loads((ROOT/'RESULTS.json').read_text())['rows']
    assert table_rows(rows) == (ROOT/'table_rows.tex').read_text()
    print(json.dumps({'status':'PASS_REDUCED_DECISION_UTILITY','comparisons':len(rows),'source_rows':len(json.loads((ROOT/'INPUTS.json').read_text())['rows'])}))
