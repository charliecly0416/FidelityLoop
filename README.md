# FidelityLoop research artifact

The release includes the [paper](paper/main.pdf), [supplement](paper/supplement.pdf),
lightweight [result checkers](paper/README.md), and an optional [CPU native replay](cpu_replay/README.md).
The CPU entry point regenerates the existing 12 Vidur-backed policy predictions;
these retrospective predictions remain separate from the original corpus below.

This corpus pairs frozen replay predictions with recorded physical deployments from seven campaigns. It supports inspection and recomputation of request timing, deadline feasibility, control targets, and scenario resource/cost accounting. It is a release of existing experiments, not an additional experiment or a workload-generalization benchmark.

| Study (archive ID) | Technically accepted physical runs | Unique replay predictions | Model/run pairs | Separate excluded attempts |
|---|---:|---:|---:|---:|
| Held-out fidelity (V3) | 36 | 60 | 180 | 2 |
| Mixed-policy (Bridge) | 38 | 28 | 76 | 0 |
| Policy deployment (E2) | 32 | 24 | 64 | 4 |
| Checkpoint screening (E3) | 26 | 28 | 52 | 0 |
| Serving-engine transfer (E4) | 26 | 42 | 78 | 2 |
| Arrival transfer (E5) | 27 | 45 | 81 | 0 |
| 14B model transfer (E6) | 12 | 12 | 24 | 0 |
| Total | 197 | 239 | 555 | 8 |

Technically accepted does not mean deadline-feasible: all 41 physical runs that fail strict P1+ remain included (V3: 3; Bridge: 20; E2: 12; E4: 2; E6: 4). Repeats reuse predictions; 555 pairs are not independent experiments. The eight aborted/technically failed attempts have inventory entries, not full usable trace records, and do not enter scientific metrics. This release covers these seven campaigns, not every supplementary experiment in the paper.

The paper uses descriptive display names; raw records and commands retain
the original archive IDs used below. [DISPLAY_NAMES.json](paper/DISPLAY_NAMES.json)
maps models, studies and checkpoints without changing their identities.

## Quantized deployment validation

The [AWQ-Marlin artifact](paper/artifact/awq_formal/README.md) contains 12 physical
runs and 18 frozen predictions, separate from the 197/239/555 corpus above.
The three-predictor comparison uses eight policy runs; four capacity anchors
are excluded. Run `python -B paper/artifact/awq_formal/verify.py` to recompute
feasibility agreement and resource/cost errors from the released evidence.

## Run locally

Python 3.10 or newer and its standard library are sufficient. No packages, GPU, API, weights, original prompts, or network are required. From this directory:

```text
python -B verify.py
python -B -m unittest -v test_trace test_e4 test_e5_e6
python -B load_trace.py
python -B evaluate.py --metric feasibility
python -B evaluate.py --metric cost
```

`verify.py` checks file hashes, the published field schema, every record's deadline/resource/cost arithmetic, pair mappings, and the saved example summary. Expected result: `PASS_PAIRED_TRACE_CORPUS`, 197 physical runs, 239 unique predictions, 555 pairs. `load_trace.py RECORD_ID` inspects one record; IDs are listed in `INDEX.json`. `--root PATH` supports relocating the complete corpus. Use `-B` to avoid bytecode caches; verification treats only listed files as release contents.

The examples report P1+ direction agreement and cost MAE separately per campaign and replay model. They use all accepted physical cells except E3/E4/E5/E6 capacity anchors, matching the paper’s per-campaign policy comparisons. Bridge/E2 include anchors. V3 examples cover all four policies, whereas the main paper's fixed-H fidelity analysis uses nine H runs. V3's original replay `feasible` flag used a different offline threshold; it is not reused as the strict P1/P1+ outcome here. In E3, six rejected *cells* under C correspond to three rejected *candidates* on steady; candidate acceptance requires both windows and physical feasibility all four repeats.

## What can be reused

Join records through the explicit pair index. Compare per-request timing and deadline decisions, examine target/state trajectories, or evaluate alternative error metrics on the recorded predictions. Additional model predictions can be mapped to the same request schedules and scored externally. With prompts and weights absent, this corpus alone cannot produce new real-serving executions, rerun language models, or retrain/evaluate a complete policy in the original closed loop.

`SCHEMA.md` specifies the projected fields and accounting. `SUMMARY.json` contains the example outputs. `INDEX.json` binds each released gzip to its SHA256 and to opaque hashes of original source components; the source components and full machine journals are retained separately. These new release hashes are not historical preregistration evidence.

The projection deliberately removes prompt text/token IDs, generated answers, source-row identities, wall-clock timestamps, hosts, process/device identifiers, environment dumps, and deployment paths. It retains relative event time, token budgets, device slots, lifecycle generations, and request aliases needed for analysis. Identity alignment is guaranteed within each declared physical/prediction pair; equal aliases across campaigns do not establish session independence or interchangeability.

Costs use the declared synthetic-API ledger, not provider bills or physical energy. Runtime verification is inherited from the accepted historical runs; the portable checker does not re-prove GPU release from OS monitors. E3's windows used separate matched hosts; its external Git anchor followed execution start. The corpus makes those deployments inspectable without isolating an environment-change causal effect.

Code and accompanying documentation use the included MIT license. The paired corpus omits model weights, original workloads and prompts. The optional CPU runner bundles pinned third-party source under its included licenses. Cite the FidelityLoop paper when using these measurements.

## SGLang transfer case study (E4)

E4 adds 26 accepted SGLang executions and 42 unique predictions from C, C30, and C_sg. The 24 non-anchor executions contain 22 strict successes and two valid H failures; both failures remain in the public data. C agrees on 16/24 cells, with zero false acceptances and eight false rejections; C30 and C_sg agree on 24/24. The measured total scenario-cost MAEs are approximately 0.5420, 0.0468, and 0.0427. `evaluate.py` reconstructs these values from request/lifecycle/ledger primitives, not copied outcome summaries.

Two C1 recovery attempts missed a controller policy-tick deadline and were repeated with the same configuration. Their anonymous inventory links to the two replacement executions. Root cause remains unresolved; excluded incomplete attempts are not scientific failures or extra comparison cells.

`E4_MODELS.json` records the backend configuration and numerical replay models. Only startup duration differs among the three replay models; their service lookup, setup, shutdown, and the deployed guard estimator are unchanged. This release is compatible with the existing version-1 trace schema and preserves every earlier compressed trace byte-for-byte. E4 uses selected known windows and checkpoints; backend and dependency changes are not isolated causal interventions. C30 and C_sg tie on classification.

## Worked screening example

Run `python -B worked_example.py --campaign E6` to recompute screening decisions, equal-window resource/cost errors, and the missed request IDs behind false accepts. S14 agrees on 4/8 policy runs with four false accepts; L14 agrees on 8/8. Each model contributes four unique predictions reused across two repeats. The example excludes capacity anchors, validates paired request schedules/accounting contracts, and retains failed requests.

`python -B worked_example.py --campaign E5` compares C/C30/C_sg on 24 policy runs each (12 unique predictions per model). All physical runs are feasible: false-accept rates are `null`, not zero-percent evidence of safe screening.

E5 uses additional constructed arrivals with fixed SGLang/7B calibration; E6 uses the 14B/vLLM deployment with measured service timing and a startup-only predictor contrast. Their original frozen predictions are included. The later T30-14 diagnostic is retrospective and is supplied separately with the paper, outside these frozen-prediction counts. These tools re-evaluate recorded deployments; they do not run a new policy, train PPO, or reproduce language-model inference.
