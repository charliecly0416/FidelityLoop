# Fixed-event decision-price sensitivity

This standalone, standard-library artifact extends the retrospective E2--E6 decision-utility case study. Run:

```text
python verify.py
python -m unittest discover -s . -p test_verify.py -v
```

INPUTS.json contains the same 234 non-anchor records as the nominal analysis, with per-component scenario costs reconstructed from checksum-bound public event traces. It contains aggregates, not a second copy of the full logs. Each source record ID/SHA and the corpus INDEX SHA are bound. NOMINAL_BASELINE.json retains the 12 original decision rows; verification requires exact recovery at nominal GPU/API prices.

For independent raw-source reconstruction, download the corpus from https://anonymous.4open.science/api/repo/FidelityLoop-144B/files and run `python source_verify.py /path/to/corpus`. This checks every component using GPU occupation intervals, lifecycle/API events, request token budgets and strict deadline outcomes, and checks complete inclusion of every released E2--E6 non-anchor record. No price component is inferred by subtracting totals.

Each GPU/API multiplier takes values 0.5, 1 and 2: all nine Cartesian combinations are reported, yielding 108 campaign/model decisions. Startup, shutdown and offline deadline-miss prices remain fixed. At every point, predicted P1+ (all requests complete by their deadlines) must hold in every window; selection then minimizes equally weighted per-window predicted cost. Exact equal-cost minimizers at released floating-point precision are all retained; the printed single selection uses original policy-ID order. Physical outcomes do not enter selection. Evaluation requires every physical repeat in every window to pass; failed selections receive no feasible-cost gap. Physical cost is first averaged across repeats within a window, then across windows; the comparator is recalculated among the tested strictly feasible candidates at each price point.

RESULTS.json preserves all 108 decisions, all minimizing sets and their feasibility, and 12 full-grid summaries; RESULTS.csv is a flat reading aid. E5/C_sg and E6/S14 tie H with H+guard throughout, so a tie-resolved single selection must not be described as inevitable. E2/C30 has a unique failing minimum throughout. Grid ranges are not confidence intervals.

This is retrospective repricing of fixed events, not new executions, policy retraining, commercial invoices or model-quality equivalence. It establishes decision consequences within the recorded pools and stated prices, not a new optimization algorithm or universal policy ranking.
