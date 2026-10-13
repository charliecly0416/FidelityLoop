# E6 startup diagnostic (CPU only)

Run with Python 3.9 or newer, using only the standard library:

```text
python verify.py
python -m unittest test_verify.py
```

The verifier recomputes feasibility classification and scalar resource/cost
errors from the reduced prediction/physical pairs, then checks equal-window
summaries and the unchanged original S14/L14 results. Each model has four
distinct policy/window predictions shared across eight physical policy runs.
Capacity anchors are separate: two predictions per model and four physical runs.
T30-14 is a post-hoc, startup-only 30-second contrast. Its service, setup,
shutdown, guard model and guard safety parameters remain those of E6; it is not
an execution-before-measurement prediction.

`POLICY_PAIRS.csv` and `ANCHOR_PAIRS.csv` preserve all paired outcomes.
`MODEL_COMPARISON.csv` and `PER_WINDOW.csv` contain the recomputable summaries.
`CONTROL_TRANSITIONS.csv` is the compact control chain, including every target
change and selected explanatory ticks. `FIRST_DIVERGENCES.csv` reports first
differences in once-per-second observations and actions, not equality of every
continuous-time physical event. `MISSED_REQUEST_CHAINS.csv` follows every request
missed by any predictor or physical run across the other predictors and repeats.
`GUARD_ACTIVITY.csv` retains active and inactive guard cases.

These records contain no prompts or host paths. SHA256 source bindings identify
the archived evidence without redistributing its deployment metadata. Request
IDs are opaque identifiers. No model weights or GPU are needed here.

This reduced artifact does **not** generate new policy predictions, rerun
inference, or reconstruct trajectory L1 from raw lifecycle logs. Trajectory L1
values are bound exported measurements; their equal-window aggregation is
checked. Target-disagreement fractions are recomputed from exported tick counts.
Full raw-event and CPU replay diagnostics are preserved in the underlying study.

Observed result: S14, T30-14 and L14 agree with 4/8, 6/8 and 8/8 policy runs,
respectively. The four successful and four failed physical runs provide the
separate false-reject and false-accept denominators. This is a two-window
diagnostic, not a startup parameter search or a general safety guarantee.
