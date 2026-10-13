"""Refresh reader-facing copies after running the original artifact verifiers."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'DISPLAY_NAMES.json').read_text())
lookup=m['models']|m['policies']|m['studies']
pattern=re.compile(r'(?<![\w\\])(?:'+'|'.join(re.escape(k) for k in sorted(lookup,key=len,reverse=True))+r')(?![\w])')
for name in ('decision_utility','decision_price'):
    src=(ROOT/'artifact'/name/'table_rows.tex').read_text().replace(r'C$_{\rm sg}$','Lookup-SG')
    dst=pattern.sub(lambda x:lookup[x.group()],src)
    (ROOT/'generated'/f'{name}_rows.tex').write_text(dst)
print('Refreshed two display tables; original artifacts unchanged.')
