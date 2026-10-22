# Prospective policy selection with generated HTTP responses

Run `python -B artifact/realapi_prospective/verify.py` from the paper root.
Only the Python standard library is required. Allow roughly 400 MB of temporary
disk space; the sealed evidence is extracted temporarily and removed on exit.
No network, model weights, Torch, or GPU is used.

`evidence.zip` contains the portable anonymous records received after the study:
36 physical execution records, 64 frozen candidate predictions, selected sets,
128 paired endpoint responses, scoring inputs, and their original verifier and
SHA256 inventory. It includes all planned requests, both H failures, all six PPO
candidates, and all four physical capacity references. The eight reference
predictions and full native GPU journals are outside this portable release.
Token IDs and generated responses are retained for HTTP payload/response and
usage checks; the quality questions are a licensed public dataset. Original
response bytes must remain intact to verify their hashes.

The wrapper first runs the sealed verifier, then independently aggregates the
36-cell matrix, physical feasibility and costs, prediction errors, and selected
policy premiums. `results.json` records the checked output. The primary cost
includes both fixed provider GPUs throughout every run. A separate client
scenario is never added to it. See Supplement Section R for the protocol and
limitations. This is verification of recorded execution, not a new inference run
or independent authentication of the experimental hardware.

Two elastic local replicas and two fixed HTTP replicas share one four-A800
host. The study reuses two known windows and existing frozen policies. Local,
30-second startup, and the timer diagnostic select H+guard. The result supports
prospective selection under actual response generation; it does not establish
that HTTP timing or host calibration is necessary to choose this winner, nor
commercial API latency, billing, or cross-model output equivalence.
