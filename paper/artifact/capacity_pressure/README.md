# Two/four-replica offline-pressure audit

Run `python -B verify.py` from this directory with Python 3.10 or newer.
The standard-library checker uses all 12 physical executions and 24 frozen
predictions from the final same-host A800 pressure study.

## Input construction

Online requests are preserved. Offline rows in `[300,1500)` are duplicated
12 times; N4 then doubles each semantic N2 row, for 1,358/2,716 planned
requests at N2/N4. Arrivals, token budgets, and deadlines are unchanged.
Prior frozen calibration selected this workload for capacity exposure before
the formal predictor-accuracy results were available.

## Records and checks

Row position identifies each request. Physical rows contain `[completion or
null, route or null, dispatch or null]`; generation rows contain `[GPU,
generation, launch, ready, shutdown, verified release]`. Times are relative to
observation start, with occupied resources clipped to `[0,2100]`. Prompts,
token IDs, machine identifiers, and original paths are omitted.

All 12 executions are technically valid: 11 pass and one fails strict joint
deadlines. Late and unfinished requests stay in the denominator. Scenario
operating costs exclude setup, cleanup, and miss penalties.

Active intervals and dispatch times establish extra-card service after 300 s
and after a device's second or later launch. All four N4 dynamic executions
use both extra GPUs after restart; setup and always-on anchors are excluded
from this dynamic-service evidence.

Selection requires predicted feasibility on this single window, then minimum
operating cost, keeping all ties within `1e-9`. The earlier short-window audit
uses a different, two-window rule. Repeat 1 is primary; repeat 2 checks
robustness using the same predictions.

The checker recomputes request, lifecycle, cost, exposure, and selection
arithmetic. GPU inference, native worker identity, and original source
authentication require the separately retained execution receipts.
