# FidelityLoop v9 paper source

Build `main.tex` or `supplement.tex` with pdfLaTeX and BibTeX. Import this ZIP
into Overleaf and select the entry point; all required figures, tables, styles,
and references are included. No shell escape or external downloads are needed.

This source integrates Vidur and the AWQ-Marlin deployment-context study into
the existing closed-loop validation argument. The AWQ extension adds 12 physical
cells while keeping the policies, windows, deadlines, and accounting contract
fixed; it is reported as a quantized-context validation, not a quantization-only
causal comparison. The main body remains 10 pages, references start on page 11,
and the main PDF has 12 pages. The supplement documents implementation and
evidence details.

The opening and conclusion explain how developers can use the framework to
diagnose replay/deployment disagreements. The background and PPO introduction
remain concise; the AWQ extension is the only new physical campaign.

The source includes lightweight, standard-library comparison artifacts.
The reduced AWQ-Marlin receipt is in `artifact/awq_formal/`; model weights and
full GPU journals remain in the separately archived return package:

- `python -B artifact/awq_formal/verify.py` reproduces the three frozen
  AWQ startup-predictor comparisons on all eight policy runs, retaining both
  deadline failures. Four regression tests check evidence tampering and strict
  deadline semantics.

- `python -B artifact/vidur_control/verify.py` recomputes deadline counts,
  occupied resources and control-target agreement for 12 predictions and
  36 physical comparisons. Four corruption/verification tests accompany it.

- `python -B artifact/vidur_service_screen/verify.py` recomputes all five
  fixed service-screen metrics from 90 reduced per-request records. This
  separate microbenchmark is not part of the 197 policy-run corpus.

- `artifact/vidur_cpu/README.md` gives the two native-output evaluations and
  nine adapter tests, plus an optional pinned-source Vidur rerun recipe.

- `python -B artifact/decision_price/verify.py` checks all nine GPU/API price
  combinations and recovers the nominal deployment choices.
- `python -B artifact/decision_utility/verify.py` reproduces deployment choices
  and observed consequences within the recorded E2–E6 candidate pools.
- `python -B artifact/e5_e6/verify.py` reproduces the original E5/E6 comparison
  table and the startup-only predictor contrast.
- `python -B artifact/e6_diagnostic/verify.py` checks the retrospective 30-second
  startup contrast and reduced event-chain evidence.
- `python -B artifact/azure_diagnostic/verify.py` checks the retrospective Azure
  comparison against strict per-request deadline feasibility.
- `artifact/e4_startup_sensitivity/recompute.py` retains the prior sensitivity
  analysis.

These lightweight checkers validate reduced evidence; they do not reproduce
GPU inference or independently reconstruct the full machine journals. The later
30-second, Azure and Vidur control analyses are retrospective and are separate from the
original frozen-prediction corpus. An optional companion `cpu_replay/` package regenerates the 12 fixed Vidur-backed
predictions using bundled source, measured profiles and pinned CPU dependencies.
It is distributed separately from this Overleaf ZIP at the anonymous release
address below. Full GPU archives and model weights are not part of this ZIP.

The companion paired-trace corpus covers the held-out study, Bridge, E2, E3,
E4, E5, and E6: 197 original physical runs, 239 original frozen predictions
and 555 pairs. The AWQ extension is a separate 12-cell deployment-context
archive and is not included in that historical corpus count. All 41 strict deadline failures remain included. Repeats reuse
predictions; the pairs are not independent experiments. The anonymous release
address is https://anonymous.4open.science/r/FidelityLoop-144B/.

`SHA256SUMS.txt` covers the source files. This build retains the MLSys 2025
style and a ten-page body budget; check the target-year submission instructions
when uploading.

Display names are unified across the paper and figures. `DISPLAY_NAMES.json`
and the supplement map these names to the unchanged archived IDs. After running
the original artifact table generators, run `python refresh_display_tables.py`
to update the two reader-facing table copies.
