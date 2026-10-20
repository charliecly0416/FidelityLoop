# FidelityLoop research artifact

FidelityLoop checks replay predictions against physical elastic-serving deployments
using request identities, deadline outcomes, control traces, and a shared cost ledger.

Start with the [paper](paper/main.pdf), [supplement](paper/supplement.pdf), or
[result-checking instructions](paper/README.md). The [CPU replay package](cpu_replay/README.md)
regenerates the existing 12 Vidur-backed policy predictions.

## Experiments and entry points

| Evidence | Recorded physical executions | Entry point |
|---|---:|---|
| Original seven-campaign paired corpus | 197 | [Schema](SCHEMA.md), `python -B verify.py` |
| Quantized AWQ-Marlin context | 12 | [AWQ audit](paper/artifact/awq_formal/README.md) |
| Continuous full-day arrivals | 8 | [Full-day audit](paper/artifact/full_day/README.md) |
| Two/four-replica short windows | 24 | [Capacity audit](paper/artifact/capacity_transfer/README.md) |
| Two/four-replica offline pressure | 12 | [Pressure audit](paper/artifact/capacity_pressure/README.md) |

The full-day and capacity studies supply the 44 additional executions described in
the paper. They retain their own inputs, pairing rules, and comparison populations;
they do not change the original 197-run corpus counts. The 12 AWQ executions are
also separate. All recorded deadline failures remain in their respective studies.

The result checkers use Python 3.10 or newer and the standard library. They require
no GPU, model weights, original prompts, external API, or network access. Run each
command from this repository root:

```text
python -B verify.py
python -B -m unittest -v test_trace test_e4 test_e5_e6
python -B paper/artifact/full_day/tools/recompute.py
python -B -m unittest discover -s paper/artifact/full_day/tests -v
python -B paper/artifact/capacity_transfer/verify.py
python -B paper/artifact/capacity_pressure/verify.py
```

These commands recompute recorded outcomes. GPU inference and new policy training
require the serving and training environment described in the paper.

## Source layout

`load_trace.py` reads the paired records, `evaluate.py` computes the metrics,
and `verify.py` checks them against the released inventory and summaries.
The checkers under `paper/artifact/` audit individual studies from their local
evidence files; each can run independently. The two capacity checkers share
accounting rules but enforce different workload and device-exposure contracts.

For the runnable control loop, start with `cpu_replay/src/controlled_vidur.py`
and its service adapter, `service_predictor.py`. The pinned implementations
under `cpu_replay/vendor/` retain their source provenance and licenses.

## Original paired corpus

The corpus contains frozen predictions and physical traces from seven campaigns:

| Study (archive ID) | Accepted physical runs | Unique predictions | Model/run pairs | Excluded attempts |
|---|---:|---:|---:|---:|
| Held-out fidelity (V3) | 36 | 60 | 180 | 2 |
| Mixed-policy (Bridge) | 38 | 28 | 76 | 0 |
| Policy deployment (E2) | 32 | 24 | 64 | 4 |
| Checkpoint screening (E3) | 26 | 28 | 52 | 0 |
| Serving-engine transfer (E4) | 26 | 42 | 78 | 2 |
| Arrival transfer (E5) | 27 | 45 | 81 | 0 |
| 14B model transfer (E6) | 12 | 12 | 24 | 0 |
| Total | 197 | 239 | 555 | 8 |

Accepted means technically valid. All 41 strict P1+ failures remain included
(V3: 3; Bridge: 20; E2: 12; E4: 2; E6: 4). Repeats reuse predictions, so 555
pairs are not independent experiments. The eight aborted or technically failed
attempts have inventory entries and are excluded from scientific metrics.

`verify.py` checks the file inventory, field schema, request deadlines, resource
and cost arithmetic, pair mappings, and saved summaries. The expected status is
`PASS_PAIRED_TRACE_CORPUS`, with 197 physical runs, 239 predictions, and 555 pairs.
Use `python -B load_trace.py RECORD_ID` to inspect an entry in `INDEX.json`,
or `python -B evaluate.py --metric feasibility` and `--metric cost` for summaries.
`--root PATH` relocates the corpus. The root manifest binds the full snapshot;
`INDEX.json` defines the historical population, whose counts remain unchanged.

The examples exclude E3/E4/E5/E6 capacity anchors; Bridge/E2 retain their anchors.
V3 examples include all four policies, while the main fixed-H analysis uses nine
H runs. Strict P1/P1+ is recomputed because V3's archived `feasible` field used
a different offline threshold. In E3, six rejected cells represent three
candidates on steady; candidate acceptance requires both windows and four
physical repeats.

[DISPLAY_NAMES.json](paper/DISPLAY_NAMES.json) maps paper names to unchanged
archive IDs. `SCHEMA.md` defines the fields. `INDEX.json` binds each compressed
record to its SHA256 and opaque source hashes. Those release hashes are not
historical preregistration evidence.

### Transfer checks

E4 contains 26 SGLang executions and 42 predictions from C, C30, and C_sg.
Its 24 non-anchor runs include 22 strict successes and two H failures. C agrees
on 16/24, with eight false rejections and no false acceptance; C30 and C_sg agree
on 24/24. Cost MAEs are approximately 0.5420, 0.0468, and 0.0427 scenario USD.
`evaluate.py` recomputes these from request and lifecycle primitives.

Two recovery attempts missed a controller tick deadline and were repeated with
the same configuration. Their inventory links to the replacements; the root
cause remains unresolved. `E4_MODELS.json` records the startup-only predictor
contrast with unchanged service lookup, setup, shutdown, and guard estimator.
Backend and dependency changes are not isolated causal interventions. Earlier
compressed records remain byte-for-byte unchanged.

`python -B worked_example.py --campaign E6` reconstructs screening, resource and
cost errors, and missed request IDs. S14 agrees on 4/8 policy runs with four
false accepts; L14 agrees on 8/8. Four unique predictions per model are reused
across two repeats; capacity anchors are excluded. The worked example checks
paired schedules and accounting and retains unfinished work.

`python -B worked_example.py --campaign E5` compares C/C30/C_sg on 24 policy runs
each, with 12 unique predictions per model. All are physically feasible, so a
false-accept rate is undefined and reported as `null`. E5 uses constructed
arrivals with fixed SGLang/7B calibration. E6 uses 14B/vLLM, measured service
timing, and a startup-only contrast. The later 30-second diagnostic and the
Vidur predictions are retrospective, outside the frozen-corpus counts.

## Reuse, anonymization, and accounting

Use the explicit pair index to compare request timing, deadline classifications,
target trajectories, and resource metrics. Additional predictions can be mapped
to the same schedules and scored externally. The records omit prompts and
weights; rerunning inference or retraining a policy requires those inputs.

Projection removes prompt text and token IDs, generated answers, source-row
identities, wall clocks, hosts, process/device identifiers, environment dumps,
and deployment paths. Relative event times, token budgets, device slots,
lifecycle generations, and within-pair request aliases remain. Equal aliases
across campaigns do not establish session independence or interchangeability.
Machine journals and original source bindings are retained separately.

The costs follow the declared synthetic-API scenario ledger, not provider bills
or physical energy. The portable checkers retain release-verification flags but
cannot re-prove GPU release from OS monitors. E3 used separate matched hosts
and its external Git anchor followed execution start; these records do not
isolate a causal effect of the environment change.

Code and documentation use the included MIT license. Bundled third-party code
keeps its own licenses. The full-day directory includes the BurstGPT attribution
and CC BY 4.0 license. Model weights and original prompts are not redistributed.
