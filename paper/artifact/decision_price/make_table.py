"""Render the complete 12-group price-grid summary for the supplement."""
import json
from pathlib import Path
from verify import run, summarize
ROOT = Path(__file__).resolve().parent

def render():
    summary = summarize(run(json.loads((ROOT/'INPUTS.json').read_text())['rows']))
    labels = {'ppo_C30_raw':'C30-2', 'ppo_C_r2':'C-2', 'ppo_C_s101':'C-1',
              'C30_s105_e0065':'N1', 'C30_s105_e0035':'N2',
              'hysteresis_u8_d0_c60':'H', 'hysteresis_u8_d0_c60__guard':'H+guard'}
    lines=[]
    for row in summary:
        model = r'C$_{\rm sg}$' if row['model']=='C_sg' else row['model']
        policies = ', '.join(labels[p] for p in row['selected_policies'])
        if row['tied_minimizer_cells']:
            policies += r'$^{\dagger}$'
        lo, hi = row['feasible_selected_gap_pct_min'], row['feasible_selected_gap_pct_max']
        gap = '--' if lo is None else (f'{lo:.2f}' if lo==hi else f'{lo:.2f}--{hi:.2f}')
        lines.append(f"{row['campaign']} & {model} & {policies} & {row['feasible_selected_cells']}/9 & {gap} " + r'\\')
    return '\n'.join(lines)+r'\bottomrule'+'\n'

if __name__ == '__main__':
    import sys
    output = ROOT/'table_rows.tex'
    if '--check' in sys.argv:
        assert output.read_text() == render()
        print('PASS_COMPLETE_PRICE_TABLE')
    else:
        output.write_text(render())
