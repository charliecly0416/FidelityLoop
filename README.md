# FidelityLoop paired traces

This corpus pairs frozen replay predictions with recorded physical deployments from four campaigns. It supports inspection and recomputation of request timing, deadline feasibility, control targets, and scenario resource/cost accounting. It is a release of existing experiments, not an additional experiment or a workload-generalization benchmark.

| Campaign | Technically accepted physical runs | Unique replay predictions | Model/run pairs | Separate excluded attempts |
|---|---:|---:|---:|---:|
| V3 | 36 | 60 | 180 | 2 |
| Bridge | 38 | 28 | 76 | 0 |
| E2 | 32 | 24 | 64 | 4 |
| E3 | 26 | 28 | 52 | 0 |
| Total | 132 | 140 | 372 | 6 |

Technically accepted does not mean deadline-feasible: all 35 physical runs that fail strict P1+ remain included (V3: 3; Bridge: 20; E2: 12). Repeats reuse predictions; 372 pairs are not independent experiments. The six aborted/technically failed attempts have inventory entries, not full usable trace records, and do not enter scientific metrics. This release covers these four campaigns, not every supplementary experiment in the paper.

## Run locally

Python 3.10 or newer and its standard library are sufficient. No packages, GPU, API, weights, original prompts, or network are required. From this directory:

```text
python -B verify.py
python -B -m unittest -v test_trace
python -B load_trace.py
python -B evaluate.py --metric feasibility
python -B evaluate.py --metric cost
```

`verify.py` checks file hashes, the published field schema, every record's deadline/resource/cost arithmetic, pair mappings, and the saved example summary. Expected result: `PASS_PAIRED_TRACE_CORPUS`, 132 physical runs, 140 unique predictions, 372 pairs. `load_trace.py RECORD_ID` inspects one record; IDs are listed in `INDEX.json`. `--root PATH` supports relocating the complete corpus. Use `-B` to avoid bytecode caches; verification treats only listed files as release contents.

The examples report P1+ direction agreement and cost MAE separately per campaign and replay model. They use all accepted physical cells except E3's two anchors, matching E3's frozen 24-cell comparison. Bridge/E2 include anchors. V3 examples cover all four policies, whereas the main paper's fixed-H fidelity analysis uses nine H runs. V3's original replay `feasible` flag used a different offline threshold; it is not reused as the strict P1/P1+ outcome here. In E3, six rejected *cells* under C correspond to three rejected *candidates* on steady; candidate acceptance requires both windows and physical feasibility all four repeats.

## What can be reused

Join records through the explicit pair index. Compare per-request timing and deadline decisions, examine target/state trajectories, or evaluate alternative error metrics on the recorded predictions. Additional model predictions can be mapped to the same request schedules and scored externally. With prompts and weights absent, this corpus alone cannot produce new real-serving executions, rerun language models, or retrain/evaluate a complete policy in the original closed loop.

`SCHEMA.md` specifies the projected fields and accounting. `SUMMARY.json` contains the example outputs. `INDEX.json` binds each released gzip to its SHA256 and to opaque hashes of original source components; the source components and full machine journals are retained separately. These new release hashes are not historical preregistration evidence.

The projection deliberately removes prompt text/token IDs, generated answers, source-row identities, wall-clock timestamps, hosts, process/device identifiers, environment dumps, and deployment paths. It retains relative event time, token budgets, device slots, lifecycle generations, and request aliases needed for analysis. Identity alignment is guaranteed within each declared physical/prediction pair; equal aliases across campaigns do not establish session independence or interchangeability.

Costs use the declared synthetic-API ledger, not provider bills or physical energy. Runtime verification is inherited from the accepted historical runs; the portable checker does not re-prove GPU release from OS monitors. E3's windows used separate matched hosts; its external Git anchor followed execution start. The corpus makes those deployments inspectable without isolating an environment-change causal effect.

Code and accompanying documentation use the included MIT license. Model weights, original workloads, prompts, and third-party source text are not redistributed. Cite the FidelityLoop paper when using these measurements.
