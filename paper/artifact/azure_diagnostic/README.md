# Azure retrospective evaluator diagnostic

Run `python -B verify.py` with Python 3.9+ and its standard library. The script takes well under one second and works without historical repositories, model weights, GPU, or network access.

This package contains all 24 fresh physical runs from the historical 0.50-load Azure-arrival comparison (four policies, two windows, three repeats), 40 **retrospective** predictions (five archived models, four policies, two windows), and all 120 explicit prediction/run pairs. CPU predictions were generated on 2026-10-12 for runs already observed in September; they were not frozen before physical execution. Each unique prediction is reused across three physical repetitions.

The verifier reconstructs strict P1+ labels from the supplied all-arrival counts, joins by the exact policy/window/run identities, and recomputes false accepts, false rejects, occupancy and scenario-cost errors, and the mixed physical outcomes of H in both windows. It checks stored pair and summary rows against these calculations and checks reduced-file hashes. P1+ requires **all online and offline requests on time**. `contract_feasible` retains the older 99%/95% qualification separately.

These are reduced counts and metrics exported after a separate author-side replay and raw-event accounting audit. **This script does not reverify raw request deadlines, lifecycle accounting, run the simulator, or rerun inference.** `SOURCE_BINDINGS.json` records the original input/log/code digests for provenance; digest strings alone do not provide the original evidence. Historical model and fixed physical-guard parameters are included for interpretation.

The Azure source supplies arrival times; request payloads use inherited token templates. The windows are already observed and capacity-conditioned at 0.50 load. This is a second-arrival-source diagnostic, not a new prompt-distribution or prospective transfer study. Costs use the declared scenario prices, not measured bills.

Files:

- `PHYSICAL_ROWS.csv`: all 24 physical runs, strict counts, resources and costs.
- `PREDICTION_ROWS.csv`: all 40 unique retrospective predictions.
- `PAIRED_ROWS.csv`: all 120 pairs, with no majority vote over repetitions.
- `SUMMARY.csv`: all-policy and dynamic-policy summaries for all five models.
- `HISTORICAL_MODELS.json`, `PARAMETERS.json`: archived predictor bank and fixed execution/guard parameters.
- `SOURCE_BINDINGS.json`: anonymous original-file digests and reduced-file hashes.
- `table_rows.tex`: the five-model, all-policy summary, generated from the reduced records.
