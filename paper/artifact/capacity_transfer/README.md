# Two/four-replica short-window audit

Run `python -B verify.py` from this directory with Python 3.10 or newer.
The standard-library checker uses all 24 physical executions and 48 frozen
predictions from the same-host A800 capacity study.

The two-replica inputs are the previously studied short windows. Four-replica
inputs duplicate each row with a new request identity; arrival times, token
budgets, and deadlines are unchanged. The comparison changes capacity and
input multiplicity together. In dynamic four-device runs, GPUs 0 and 1 serve
requests; always-on anchors use all four. The separate pressure study tests
dynamic service on the extra devices.

## Record fields

- Input: `[online/offline, arrival, deadline, input budget, output budget]`.
- Request: `[completion or null, route or null, dispatch or null]`. Row position
  is the request identity shared across policies.
- Generation: `[GPU, generation, launch, ready, shutdown-issued, verified-release]`.
- Physical tick: `[tick, target, active-ready, starting, online queue, offline queue]`.

Times are relative to observation start. Generation records retain setup and
cleanup; occupied time is clipped to `[0,2100]`. Predictions preserve state
transitions, request outcomes, and frozen operating costs. Prompts, token IDs,
machine identifiers, absolute paths, and wall-clock dates are omitted.

## Recomputed results and selection

All 24 executions are technically valid: 21 pass and three fail the strict
joint deadline contract. The checker retains every planned request and
recomputes feasibility, occupation, routing, costs, and candidate selection.
Costs exclude setup, cleanup, and miss penalties and use the scenario ledger.

Candidate selection requires predicted feasibility on both windows, then
minimizes mean predicted operating cost. All ties within `1e-9` are kept.
Repeat 1 is primary; repeat 2 checks robustness with the same predictions.

The records support a reduced-data audit. GPU inference and native worker
identity checks require the separately retained execution receipts.
