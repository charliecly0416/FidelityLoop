# Artifact scope

This directory contains the protocol/configuration files needed to inspect the registered contract, a 32-row derived workload sample, and `e1_repricing/`, a standalone verifier for the E1 fixed-event resource and scenario-cost calculation.

The E1 subdirectory includes its own README, license, manifest, reduced records, analysis, and tests. It can be copied or extracted independently of this repository. From the repository root, run `python -B artifact/e1_repricing/verify.py`; it checks 169 numerical results. The release also provides this directory as a standalone ZIP.

Private prompts, model weights, multi-gigabyte GPU journals, and transient development archives are excluded. Reduced E1 acceptance counts are inherited from the accepted runs; repricing does not independently validate the original GPU execution. See `REPRODUCING.md` at the repository root for the complete entry-point map.
