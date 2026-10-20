"""Verify anonymous reduced request/lifecycle records using the standard library."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "FILES_SHA256.json").read_text())
    for name, sha in manifest.items():
        assert Path(name).name == name, "unsafe member name"
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, (
            "file digest mismatch"
        )
    data = json.loads(gzip.decompress((root / "records.json.gz").read_bytes()))
    physical = {}
    labels = []
    for run in data["physical"]:
        inputs = data["inputs"][str(run["N"]) + "_" + run["window"]]
        assert len(inputs) == len(run["requests"])
        assert len(run["ticks"]) == 2100 and [r[0] for r in run["ticks"]] == list(
            range(2100)
        )
        counts = {
            kind: dict(arrivals=0, on_time=0, late=0, unfinished=0)
            for kind in ("online", "offline")
        }
        routed = Counter()
        for source, terminal in zip(inputs, run["requests"]):
            kind, arrival, deadline, tokens_in, tokens_out = source
            finished, route, dispatched = terminal
            assert deadline == arrival + ({"online": 60, "offline": 300}[kind])
            counts[kind]["arrivals"] += 1
            if isinstance(route, int):
                assert 0 <= route < run["N"]
                routed[str(route)] += 1
            if finished is None or finished > 2100:
                counts[kind]["unfinished"] += 1
            else:
                assert dispatched is not None and finished >= dispatched >= arrival
                counts[kind]["on_time" if finished <= deadline else "late"] += 1
        feasible = all(
            v["arrivals"] > 0 and v["on_time"] == v["arrivals"] for v in counts.values()
        )
        occupied = [0.0] * run["N"]
        starts = stops = 0
        per_gpu_last = [-float("inf")] * run["N"]
        for gpu, generation, start, ready, stop, end in sorted(
            run["generations"], key=lambda g: (g[0], g[2])
        ):
            assert (
                0 <= gpu < run["N"]
                and start <= ready <= stop <= end
                and start >= per_gpu_last[gpu]
            )
            per_gpu_last[gpu] = end
            occupied[gpu] += max(0.0, min(2100.0, end) - max(0.0, start))
            starts += int(0 <= start < 2100)
            stops += int(0 <= stop < 2100)
        assert all(0 <= s <= 2100 + 1e-6 for s in occupied)
        api_cost = sum(
            1e-6 * source[3] + 3e-6 * source[4]
            for source, request in zip(inputs, run["requests"])
            if request[1] == "synthetic_api"
            and request[2] is not None
            and request[2] < 2100
        )
        cost = 0.001 * sum(occupied) + 0.002 * starts + 0.001 * stops + api_cost
        expected = run["expected"]
        assert feasible == expected["P1plus"] and counts == expected["counts"], (
            "request/feasibility mismatch"
        )
        assert abs(cost - expected["cost"]) < 1e-8, "cost mismatch"
        assert abs(sum(occupied) - expected["gpu_seconds"]) < 1e-6, (
            "occupied resource mismatch"
        )
        assert {str(g): routed[str(g)] for g in range(run["N"])} == expected[
            "per_gpu_requests"
        ], "GPU routing mismatch"
        exposure_counts = Counter()
        restarted = Counter()
        for finished, route, dispatched in run["requests"]:
            if (
                type(route) is int
                and dispatched is not None
                and 300 <= dispatched < 1800
            ):
                gens = [
                    g
                    for g in run["generations"]
                    if g[0] == route and g[3] <= dispatched <= g[4]
                ]
                assert len(gens) == 1, "dispatch not bound to ready generation"
                exposure_counts[route] += 1
                if gens[0][1] >= 2:
                    restarted[route] += 1
        intervals = run["active_intervals"]
        assert all(a <= b for a, b, gs in intervals)
        assert all(
            abs(b - c) < 1e-6 for (_, b, _), (c, _, __) in zip(intervals, intervals[1:])
        )
        ready3 = sum(
            max(0, min(b, 1800) - max(a, 300)) for a, b, gs in intervals if len(gs) >= 3
        )
        for a, b, gs in intervals:
            assert len(gs) == len(set(gs)) and all(0 <= g < run["N"] for g in gs)
            for g in gs:
                assert any(
                    q[0] == g and q[3] <= a + 1e-6 and b <= q[5] + 1e-6
                    for q in run["generations"]
                ), "ready interval outside generation"
        ex = run["expected_exposure"]
        assert {str(g): exposure_counts[g] for g in range(run["N"])} == ex[
            "post300_per_gpu_requests"
        ]
        assert {str(g): restarted[g] for g in range(run["N"])} == ex[
            "post300_after_restart_requests"
        ]
        assert abs(ready3 - ex["ready_ge3_seconds"]) < 1e-6
        passed = (
            (
                all(exposure_counts[g] >= 20 and restarted[g] >= 10 for g in (2, 3))
                and ready3 >= 120
            )
            if run["N"] == 4
            else None
        )
        assert passed == ex["extra_card_exposure_pass"]
        physical[run["run_id"]] = dict(P1plus=feasible, cost=cost)
        labels.append(feasible)
    assert len(physical) == 12 and sum(labels) == 11
    predictions = {}
    for pred in data["predictions"]:
        inputs = data["inputs"][str(pred["N"]) + "_" + pred["window"]]
        assert len(inputs) == len(pred["requests"])
        feasible = all(
            r[0] is not None and r[0] <= min(2100, s[2])
            for s, r in zip(inputs, pred["requests"])
        )
        states = ["active"] * pred["N"]
        last = 0.0
        occupied = 0.0
        starts = stops = 0
        for t, gpu, previous, state in pred["states"]:
            assert 0 <= t <= 2100 and t >= last and states[gpu] == previous
            occupied += (t - last) * sum(s != "off" for s in states)
            starts += int(previous == "off" and state == "starting")
            stops += int(state == "stopping" and previous != "stopping")
            states[gpu] = state
            last = t
        occupied += (2100 - last) * sum(s != "off" for s in states)
        api_cost = sum(
            1e-6 * s[3] + 3e-6 * s[4]
            for s, r in zip(inputs, pred["requests"])
            if r[1] == "synthetic_api" and r[2] is not None and r[2] < 2100
        )
        cost = 0.001 * occupied + 0.002 * starts + 0.001 * stops + api_cost
        assert feasible == pred["P1plus"] and abs(cost - pred["cost"]) < 1e-8, (
            "prediction ledger mismatch"
        )
        predictions[(pred["N"], pred["model"], pred["window"], pred["policy"])] = dict(
            P1plus=feasible, cost=cost
        )
    assert len(predictions) == 24
    selected = []
    for n in (2, 4):
        for model in ("Share", "Lookup69", "Lookup30", "Local"):
            viable = {}
            for policy in ("H", "Hguard"):
                group = [predictions[(n, model, w, policy)] for w in ("pressure",)]
                if all(p["P1plus"] for p in group):
                    viable[policy] = sum(p["cost"] for p in group)
            best = min(viable.values()) if viable else None
            chosen = [p for p, c in viable.items() if abs(c - best) <= 1e-9]
            assert chosen == next(
                s["selected"]
                for s in data["selection"]
                if (s["N"], s["model"]) == (n, model)
            )
            selected.append(dict(N=n, model=model, selected=chosen))
    return dict(
        status="PASS",
        physical_runs=12,
        strict_passes=sum(labels),
        scientific_failures=12 - sum(labels),
        frozen_predictions=24,
        selection=selected,
        scope="Reduced-record request, lifecycle, cost, routing and frozen-selection arithmetic; not GPU re-execution.",
    )


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
