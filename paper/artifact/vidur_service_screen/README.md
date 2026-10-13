# Vidur Qwen2.5-7B service screen

Run `python -B verify.py` in this directory (Python standard library only).
The checker validates file hashes and recomputes all five error metrics and the
fixed gate from reduced per-request measurements, retaining all 36 groups and
90 requests. `request_results.csv` supplies the observed and predicted timing;
`predictions.json` and `admissions.json` retain the prediction output/input.

`mlp.csv`, `attention.csv`, and `profile_manifest.json` preserve measured
operator profiles. `protocol.json` is the original frozen service protocol;
its original case hash refers to the private full-token input, whereas
`service_cases_public.json` is explicitly a reduced public shape projection.
`SCREEN_PROTOCOL.json` binds the full return archive and runtime identities.
The full return archive contains prompt/token IDs, machine paths and profiler
traces; it is retained separately and is not part of this compact release.

This checker recomputes errors from recorded predictions; it does not refit
Vidur or rerun GPU inference. The fixed Vidur/Sarathi configuration is recorded
in `predictions.json`. `PREDICTOR_FIX.patch` records two CPU execution fixes
(group-local entity IDs and floating-point CSV comparison), neither changing
profiles, fitting parameters nor error thresholds. Vidur's MIT license is
included. The service screen is separate from the 197 policy-run corpus.

The result is PASS_SERVICE_SCREEN_ONLY: request-latency APE median/P90
18.36%/19.74%, group-makespan APE median/P90 18.36%/19.66%, and worst-condition
median 19.80%. All group predictions underestimate actual makespan; this test
does not isolate runtime-stack differences from unmodeled overhead. It does
not measure lifecycle, policy-selection, or occupied-resource fidelity.
