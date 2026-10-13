# E4 retrospective startup sensitivity

This analysis changes only the replay startup scalar and scores the resulting
predictions against the same 24 non-anchor SGLang runs. It includes 18 tested
values and the original measured-median reference. Each model/point has 12
unique policy/window predictions, reused for two physical repeats. Total cost
includes the declared offline deadline-miss penalty.

Run `python -B recompute.py` with Python 3.10 or newer. The standard-library
script reproduces agreement, false accepts/rejects, mismatched arms, and cost
MAE from the reduced per-comparison rows. It does not rerun policies, establish
new physical outcomes, or recompute cost primitives. Replaying policies requires
the separately retained frozen E4 inputs, checkpoints, and source environment.

The physical record IDs map to the existing paired trace corpus. New sensitivity
predictions are retrospective and are separate from its 182 predictions frozen
before execution; the original 450-pair population is unchanged. RESULTS.json
binds each parameter point to its retained full replay receipt. The scalar grid
is finite: no continuous tolerance interval or monotonic response is asserted.
