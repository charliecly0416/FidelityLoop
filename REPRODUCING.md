# Reproducing the portable checks

The source release and the standalone E1 artifact serve different purposes. Neither needs a GPU, API key, model weights, or the original production prompts. The full GPU campaigns remain separately archived.

## Framework checks (source release)

Use Python 3.10 or newer. From the repository root, install the local package and run:

```text
python -m pip install -e .
python experiments/run_framework_validation.py
python -m fidelityloop.framework.validate
```

The first command may install the Python build tools; the checks themselves make no network requests. If setuptools is already installed and an offline installation is needed, use `python -m pip install --no-build-isolation --no-deps -e .`.

| Command | Output | Expected evidence |
|---|---|---|
| `python experiments/run_framework_validation.py` | `results/VALIDATION.json` | PASS; three synthetic configurations, five requests each; request/state/cost conservation and PPO action mapping |
| `python -m fidelityloop.framework.validate` | `results/facade/VALIDATION.json` | PASS; four historical policy-compatibility cases and two synthetic device/slot cases, three requests per case |

The second result matches `evidence/HISTORICAL_FACADE_VALIDATION.json`, including request, event, and ledger hashes. It uses the unchanged contract in `artifact/bridge_execution/inputs/simulator_contract.json`. It compares a facade with the historical simulator under small synthetic inputs; it does not rerun the physical campaigns. An output directory can be selected with `--output PATH`.

To run the repository regression tests, install pytest if needed and run:

```text
python -m pip install pytest
python -m pytest -q
```

The five tests include the historical facade CLI and a relocated E1 artifact verification/test run.

## E1 resource and scenario-cost calculation (standalone artifact)

The directory `artifact/e1_repricing/` is self-contained. The release asset `FidelityLoop_E1_CPU_Artifact_v7r8_20261007.zip` contains the same files at its root. It requires only Python 3.9+ and the standard library.

From that directory, run:

```text
python -B verify.py
python -B -m unittest -v test_analysis
python -B analysis.py
```

Keep `-B` and write any redirected output outside the artifact directory: the verifier checks an exact inventory and rejects unexpected files, including generated caches.

Expected results:

- `PASS_REDUCED_LEDGER_REPRICING`, with 169 numerical values checked against full-precision reference results.
- Ten unit tests pass, including invalid-input, independent arithmetic, denominator, price-boundary, and manifest checks.
- Four window/accounting results from 14 reduced ledgers: 12 comparisons and two excluded capacity anchors.
- Window scenario-cost reduction of approximately 32.597479% in steady and 49.359762% in recovery; API break-even multipliers approximately 3.083750932523 and 11.990313753803 at the registered GPU price.

Only fixed-event arithmetic is recomputed. The accepted completion counts, routes, and phase measurements are inherited inputs; the verifier does not re-establish deadlines from raw GPU logs or predict a policy response to changing prices. Model quality, real API billing, and energy are outside this calculation.

## Decision reconstruction evidence

`evidence/E1_DECISION_REPLAY_AUDIT.json` binds the exact SHA256 of `E1_DECISION_REPLAY_SUMMARY.json`. These original records cover 29,580 logged ticks across 16 earlier same-runtime runs: 25,200 comparison ticks, 4,200 capacity-anchor ticks, and 180 smoke ticks. They record causal observation/state recovery and replayed decisions; their input traces are retained in the historical archive, so this compact release distributes the audit receipts rather than a full raw-event recovery runner.

This evidence is distinct from the 32-cell E2 PPO campaign. Replaying decisions under recorded events checks implementation consistency, not prediction accuracy of an unconstrained rollout.

## API-delay diagnostic receipt

`evidence/API_DELAY_SENSITIVITY.json` reports the frozen-rule CPU sweep in supplement B.3. The tested delays are 1, 2, 5, 10, 20, 40, and 60 seconds; all other settings are fixed. The full replay inputs and runner are retained separately, so the public file supports result inspection rather than a standalone rerun.

## Historical archive paths in the supplement

The supplement's `scripts/`, `docs/`, and `artifacts/` paths identify the full historical experiment archive. They are provenance locations, not commands available in this compact source release.

| Historical reference | Available here | Scope |
|---|---|---|
| `scripts/maxopt_v3_closeout/reverify.py` and `analyze.py` | Generated paper tables and source hashes | Full 36-run raw verification requires the retained archive; not replaced by a CPU smoke test |
| `tests/maxopt_framework/test_facade.py` / historical facade report | `python -m fidelityloop.framework.validate`, its regression test, and the historical receipt | The four-plus-two synthetic compatibility/conservation results can be reproduced locally |
| `docs/maxopt_v5_e1_boundary_20260930/analysis.py` | `artifact/e1_repricing/analysis.py` and `verify.py` | The same fixed-event resource/cost calculation in a portable reduced format |
| Earlier E1 decision-recovery script and logs | `evidence/E1_DECISION_REPLAY_*.json` | Unchanged audit receipts; full raw replay is not included |
| E2 physical runtime/checkpoints/raw journals | Compact E2 ledgers and prediction recheck in `evidence/` | Reported outcomes and source identities can be inspected; no claim of a turnkey GPU rerun |

## Paper builds

The Overleaf ZIP contains root-level `main.tex` and `supplement.tex`, figures, tables, bibliography, and styles. Use pdfLaTeX/BibTeX for each entry point. The paper remains anonymous; the public source repository is a separate, named distribution. The standalone E1 ZIP uses author-neutral contents and has no repository/account link in its files.
