# Vidur-backed closed-loop comparison

Run `python -B verify.py` with the Python standard library. It recomputes
deadline counts, GPU occupation, API routing counts, and tick-level target
agreement for all 12 CPU predictions and their 36 physical comparisons.

`requests.csv.gz` retains each arrival, deadline, route, dispatch, completion
and terminal status. Empty completion fields are unfinished work, not dropped
samples. `occupation_intervals.csv.gz` contains non-off intervals clipped to
the 2,100-second observation window. `targets.csv.gz` contains every one-second
decision. The verifier checks coverage, input identity and nonoverlapping
intervals, then compares recomputed metrics with `expected.json`.

CPU service is native Vidur/Sarathi using Qwen2.5-7B/A100 operator profiles;
lifecycle timing and the guard retain FidelityLoop's frozen development
parameters. This is a retrospective comparison, with no outcome fitting and
no new policy-accuracy acceptance threshold. Its physical runs already belong
to the 197-run corpus; the 12 later predictions do not change the published
239 original frozen-prediction count.

These anonymous projections contain no prompts, absolute machine clocks,
hostnames, process IDs or machine paths. `provenance.json` binds the full-archive
analysis. The verifier checks reduced evidence; it does not rerun the native
simulator, the random-forest fit or GPU inference. Profiles and isolated
service validation are in `../vidur_service_screen/`.
