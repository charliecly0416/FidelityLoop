"""S1: controller-clock lifecycle endpoints, run-balanced estimates, RAW queues."""
import statistics
import random
from collections import defaultdict

from fidelityloop.legacy.maxopt_v2.calibrated import quantile
from .common import OUT, RAW, CAMPAIGN, H_ID, read, lines, save, digest, verify_protection


def stats(values):
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                p90=quantile(values, .9), p95=quantile(values, .95), maximum=max(values))


def balanced(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['run_id']].append(row[key])
    medians = [statistics.median(v) for _, v in sorted(groups.items())]
    rng = random.Random(20260921)
    boots = [statistics.median(rng.choices(medians, k=len(medians))) for _ in range(1000)]
    return dict(value=statistics.median(medians), run_count=len(medians), generation=stats([r[key] for r in rows]),
                run_medians=medians, exploratory_run_bootstrap95=[quantile(boots,.025), quantile(boots,.975)])


def parameters(samples, run_ids=None):
    selected = [s for s in samples if run_ids is None or s['run_id'] in run_ids]
    startup = [s for s in selected if s['launch_phase'] == 'window']
    shutdown = [s for s in selected if s['shutdown_phase'] == 'normal_scale_in']
    if not startup or not shutdown:
        raise ValueError('no endpoint-consistent lifecycle support')
    return dict(startup_seconds=balanced(startup, 'startup_seconds')['value'],
                shutdown_seconds=balanced(shutdown, 'shutdown_seconds')['value'],
                startup_safe_seconds=quantile([s['startup_seconds'] for s in startup],.95),
                startup_max_seconds=max(s['startup_seconds'] for s in startup),
                startup_min_seconds=min(s['startup_seconds'] for s in startup),
                shutdown_min_seconds=min(s['shutdown_seconds'] for s in shutdown),
                startup=balanced(startup,'startup_seconds'), shutdown=balanced(shutdown,'shutdown_seconds'))


def shutdown_phase(event, origin, cutoff):
    if (origin <= event['controller_monotonic_ns'] < cutoff and not event['forced']
            and event['reason'] == 'drain_completed' and event['phase'] == 'window'):
        return 'normal_scale_in'
    if event['phase'] == 'cleanup' or event['controller_monotonic_ns'] >= cutoff:
        return 'terminal_cleanup'
    raise ValueError('unexpected shutdown context')


def queue_area(requests, end=2100):
    total = 0.
    for r in requests.values():
        stop = end if r['dispatched_at'] is None else min(end, r['dispatched_at'])
        if stop < r['arrival_s']:
            raise ValueError('dispatch before arrival')
        total += stop-r['arrival_s']
    return total


def run():
    verify_protection()
    raw, samples, barriers, bindings, truth = read(RAW), [], [], {}, {}
    for rid, data in sorted(raw.items()):
        path = CAMPAIGN / (rid + '__attempt%02d' % data['summary']['attempt']) / 'controller_events.jsonl'
        events = lines(path)
        bindings[str(path)] = digest(path)
        origin, cutoff = data['origin_ns'], data['origin_ns']+2100*10**9
        stops = {(e['gpu'],e['generation']):e for e in events if e['kind']=='shutdown_issued'}
        for g in data['generations']:
            event = stops[g['gpu'],g['generation']]
            if event['controller_monotonic_ns'] != g['shutdown']:
                raise ValueError('RAW endpoint disagreement')
            samples.append(dict(**g, run_id=rid, window=data['summary']['window'], repeat=data['summary']['repeat'],
                                shutdown_phase=shutdown_phase(event,origin,cutoff), cache_state='unknown'))
        start = next(e['controller_monotonic_ns'] for e in events if e['kind']=='setup_start')
        ready = next(e['controller_monotonic_ns'] for e in events if e['kind']=='local_ready_barrier')
        cleanup = next(e['controller_monotonic_ns'] for e in events if e['kind']=='cleanup_start')
        end = next(e['controller_monotonic_ns'] for e in events if e['kind']=='cleanup_complete')
        barriers.append(dict(run_id=rid, setup_barrier_seconds=(ready-start)/1e9, joint_cleanup_seconds=(end-cleanup)/1e9))
        truth[rid] = dict(summary=data['summary'], queue_seconds=queue_area(data['requests']),
                          targets=data['targets'], requests=data['requests'])
    p = parameters(samples)
    p.update(setup_seconds=statistics.median(b['setup_barrier_seconds'] for b in barriers),
             joint_cleanup_seconds=statistics.median(b['joint_cleanup_seconds'] for b in barriers),
             schema='maxopt-v3-lifecycle-v1', cache_identified=False,
             sampling='run-balanced median; p95 generation quantile is safety heuristic, not coverage guarantee')
    save(OUT / 's1/lifecycle_samples.json', samples)
    save(OUT / 's1/startup_model_v3.json', p)
    save(OUT / 's1/barriers.json', barriers)
    save(OUT / 's1/truth.json', truth)
    # Grid rule fixed before any closed-loop loss: same multiplicative grid with lower clipping.
    save(OUT / 's1/gate.json', dict(status='PASS', source_hashes=bindings, raw_sha256=digest(RAW),
         samples=len(samples), window_startups=sum(s['launch_phase']=='window' for s in samples),
         normal_shutdowns=sum(s['shutdown_phase']=='normal_scale_in' for s in samples),
         queue_reconstructable=True, grid_revision=False,
         grid_reason='Empirical center is within observed range; retain approved grid and clip lower boundary.',
         joint_barriers=stats([b['setup_barrier_seconds'] for b in barriers]),
         protected_files_verified=verify_protection()))
    print('S1 PASS', {k:p[k] for k in ('startup_seconds','shutdown_seconds','startup_safe_seconds')})


if __name__ == '__main__':
    run()
