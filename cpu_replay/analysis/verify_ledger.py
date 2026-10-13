"""Independent saved-event audit; standard library, no controller import."""
import argparse
import json
import math
from pathlib import Path


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(folder):
    folder = Path(folder)
    events = read_lines(folder / 'events.jsonl')
    decisions = read_lines(folder / 'decisions.jsonl')
    records = json.loads((folder / 'records.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    horizon = summary['horizon']
    assert [d['time'] for d in decisions] == list(range(horizon)), 'missing/duplicate control tick'
    devices = {}
    seen, dispatched, complete = set(), {}, {}
    queued = {'online': [], 'offline': []}
    api = set()
    state_area = {s: 0.0 for s in ['off', 'starting', 'active', 'draining', 'stopping']}
    state_at = {}
    last = 0.0
    decisions_at = {}
    for decision in decisions:
        decisions_at.setdefault(decision['event_seq_before_action'], []).append(decision)
    transitions = {
        'off': {'starting'}, 'starting': {'active'},
        'active': {'draining'}, 'draining': {'active', 'stopping'},
        'stopping': {'off'},
    }
    def check_decisions(index):
        for decision in decisions_at.get(index, []):
            obs = decision['observation']
            assert set(obs) == {'schema', 'time', 'queue_online', 'queue_offline', 'devices'}
            assert obs['time'] == decision['time'] < horizon
            for kind in queued:
                assert [r['request_id'] for r in obs['queue_' + kind]] == queued[kind]
            assert len(obs['devices']) == 2
            for gpu, d in enumerate(obs['devices']):
                assert d['state'] == devices[gpu]['state']
                close(d['since'], state_at[gpu])
                assert {r['request_id'] for r in d['jobs']} == devices[gpu]['jobs']
            visible = [r for kind in queued for r in obs['queue_' + kind]]
            visible += [r for d in obs['devices'] for r in d['jobs']]
            assert len({r['request_id'] for r in visible}) == len(visible)
            for row in visible:
                assert set(row) == {'request_id', 'job_type', 'arrival_s', 'deadline_s',
                                    'input_tokens', 'max_output_tokens', 'started_at'}
                rid = row['request_id']
                assert rid in seen and row['arrival_s'] <= decision['time']
                for field in ['job_type', 'arrival_s', 'deadline_s', 'input_tokens', 'max_output_tokens']:
                    assert row[field] == records[rid][field]
                expected_start = dispatched[rid]['time'] if rid in dispatched else None
                assert row['started_at'] == expected_start
    for i, event in enumerate(events):
        check_decisions(i)
        t, kind = event['time'], event['kind']
        assert math.isfinite(t) and last <= t <= horizon
        last = t
        if kind == 'state':
            gpu = event['device_id']
            old, new = event['previous'], event['state']
            generation = event['generation']
            if old is None:
                assert gpu not in devices and t == 0 and new in ('off', 'active')
                devices[gpu] = {'state': new, 'generation': generation, 'jobs': set()}
            else:
                d = devices[gpu]
                assert d['state'] == old and new in transitions[old]
                state_area[old] += t - state_at[gpu]
                assert generation == d['generation'] + (new == 'starting')
                if new in ('off', 'starting', 'stopping'):
                    assert not d['jobs'], ('released unfinished work', gpu, t)
                d.update(state=new, generation=generation)
            state_at[gpu] = t
        elif kind == 'arrival':
            rid = event['request_id']
            assert rid not in seen and records[rid]['arrival_s'] == t < horizon
            seen.add(rid)
            queued[records[rid]['job_type']].append(rid)
        elif kind == 'dispatch':
            rid = event['request_id']
            assert rid in seen and rid not in dispatched and t < horizon
            k = records[rid]['job_type']
            assert queued[k][0] == rid, ('non-FIFO dispatch', rid)
            queued[k].pop(0)
            assert records[rid]['dispatched_at'] == t
            dispatched[rid] = event
            if event['route'] == 'local':
                gpu = event['device_id']; d = devices[gpu]
                assert d['state'] == 'active' and len(d['jobs']) < 4
                assert event['generation'] == d['generation']
                assert records[rid]['route'] == gpu
                d['jobs'].add(rid)
            else:
                assert event['route'] == records[rid]['route'] == 'synthetic_api'
                assert k == 'online' and t - records[rid]['arrival_s'] >= 10
                assert len(api) < 8
                api.add(rid)
        elif kind == 'complete':
            rid = event['request_id']
            assert rid in dispatched and rid not in complete
            close(t, records[rid]['completed_at'])
            assert dispatched[rid]['time'] <= t
            complete[rid] = t
            if event['route'] == 'local':
                gpu = event['device_id']; d = devices[gpu]
                assert d['state'] in ('active', 'draining')
                assert event['generation'] == d['generation'] == dispatched[rid]['generation']
                assert gpu == dispatched[rid]['device_id'] and rid in d['jobs']
                d['jobs'].remove(rid)
            else:
                assert rid in api and event['route'] == 'synthetic_api'
                close(t - dispatched[rid]['time'], 5)
                api.remove(rid)
        else:
            raise AssertionError(('unexpected ledger event', kind))
    check_decisions(len(events))
    assert all(0 <= i <= len(events) for i in decisions_at)
    assert seen == {rid for rid,r in records.items() if r['arrival_s'] < horizon}
    for gpu,d in devices.items():
        state_area[d['state']] += horizon - state_at[gpu]
    close(sum(state_area.values()), 2 * horizon)
    occupied = sum(v for k,v in state_area.items() if k != 'off')
    close(occupied, summary['occupied_seconds'])
    assert [devices[i]['state'] for i in range(2)] == summary['final_states']
    assert len(api) == summary['api_pending']
    populations = {}
    for k in ['online', 'offline']:
        ids = {rid for rid in seen if records[rid]['job_type'] == k}
        done = ids & complete.keys()
        timely = {rid for rid in done if complete[rid] <= records[rid]['deadline_s']}
        populations[k] = dict(arrivals=len(ids), completed=len(done), on_time=len(timely),
                              late=len(done)-len(timely), unfinished=len(ids)-len(done))
        actual = summary['metrics']['populations'][k]
        for key, value in populations[k].items():
            assert actual[key] == value, (k,key,actual[key],value)
    assert all((r['completed_at'] is not None) == (rid in complete) for rid,r in records.items())
    return {'status': 'PASS', 'case': folder.name, 'decisions': len(decisions),
            'arrived': len(seen), 'completed': len(complete), 'occupied_seconds': occupied,
            'state_seconds': state_area, 'populations': populations}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case', type=Path)
    args = p.parse_args(); print(json.dumps(audit(args.case), indent=2))
