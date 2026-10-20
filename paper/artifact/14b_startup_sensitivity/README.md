# Retrospective 14B startup sensitivity

This CPU-only analysis varies only the 14B Local replay startup scalar and
scores each prediction against the eight retained physical policy runs. The
finite grid contains 20--90 s at 1 s and 39--50 s at 0.25 s, plus measured
reference points. The guard model, service lookup, physical outcomes, and all
other replay fields remain fixed. It is retrospective and does not add GPU
runs or establish a continuous threshold.

`grid_14b.json`, `fine_14b.json`, and `probe.json` are the exact retained
outputs. The reduced interval table in the supplement is derived from them.
The replay verifier is `recompute.py`; it reuses the frozen E6 source package
and output archive on CPU. Set `FIDELITYLOOP_E6_RECEIVED` to that package root
before running it. No frozen paper or physical result is modified.
