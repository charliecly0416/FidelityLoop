# E5/E6 reduced comparison evidence

Run `python verify.py` from this folder (or invoke the script by path).
The standard-library verifier checks the source-bound CSV hashes, recomputes
feasibility/error directions and equal-window resource/cost/trajectory errors,
checks repeated prediction identities, and compares the emitted values with
`generated/e5e6_rows.tex`. E6 S14/L14 parameters must differ only in startup.

The CSVs are exact exports from CPU-accepted analyses. A row in the per-cell
file is a model/physical-run comparison; repeated runs reuse predictions.
E5 has 24 policy runs and 12 unique policy/window predictions per model;
E6 has 8 policy runs and 4 unique predictions per model. Anchors are excluded.
E5 has no physically infeasible policy runs, so its false-accept rate is
undefined. E6 retains four failed H runs. Costs are total scenario USD,
including declared deadline penalties; resource and trajectory errors are GPU
seconds. Guard slack is queued-offline time-to-deadline, not running-job slack.

This reduced evidence supports aggregation checks. It cannot regenerate raw
physical events or inference outputs. It supplements, without altering, the
separate 158-run anonymous paired-trace corpus.
