"""Optional full-source verification: python source_verify.py /path/to/corpus."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main(corpus):
    corpus = Path(corpus).resolve()
    root = Path(__file__).resolve().parent
    inputs = json.loads((root/'INPUTS.json').read_text())
    assert hashlib.sha256((corpus/'INDEX.json').read_bytes()).hexdigest() == inputs['corpus_index_sha256']
    spec = importlib.util.spec_from_file_location('corpus_loader', corpus/'load_trace.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    index = loader.read_index(corpus)
    entries = {e['record_id']: (c['campaign'], kind, e)
               for c in index['campaigns'] for kind in ('physical', 'predictions') for e in c[kind]}
    required = {rid for rid, (campaign, kind, e) in entries.items()
                if campaign in ('E2', 'E3', 'E4', 'E5', 'E6') and e['policy'] != 'all2'}
    assert {r['record_id'] for r in inputs['rows']} == required
    assert len(inputs['rows']) == len(required)
    for row in inputs['rows']:
        campaign, kind, entry = entries[row['record_id']]
        assert campaign == row['campaign'] and kind == row['kind']
        assert entry['sha256'] == row['record_sha256']
        assert (entry['policy'], entry['window'], entry.get('replay_model')) == (row['policy'], row['window'], row['replay_model'])
        record = loader.load_trace(entry, corpus)
        metrics = loader.measure(record)
        loader.check_reference(record, metrics)
        for key in ('P1plus', 'cost_total', 'costs'):
            assert metrics[key] == row[key], (row['record_id'], key)
    print(json.dumps({'status':'PASS_FULL_SOURCE_PRICE_COMPONENTS', 'records':len(inputs['rows']),
                      'index_sha256':inputs['corpus_index_sha256']}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
