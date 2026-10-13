# Retrospective deployment-choice example

Run `python -B verify.py` and `python -B -m unittest -v test_verify` with Python 3.10+; no packages or network are needed.

`INPUTS.json` contains strict P1+ and scenario-cost aggregates reconstructed from the public E2–E6 paired corpus. The verifier recomputes selection, observed run success, equal-window cost and cost gaps, and checks the accompanying table. It verifies aggregation, not raw GPU execution or per-request ledgers; those can be checked with the full corpus at https://anonymous.4open.science/r/FidelityLoop-144B/.

One policy is selected per campaign/model: require predicted P1+ in every window, then minimize equal-window mean predicted cost. Raw policy ID breaks ties. All prediction minimizers are retained because E5 C_sg and E6 S14 contain ties. Inputs exclude all2 capacity anchors. A failed selected policy has no feasible-cost gap; its low resource use cannot count as a saving. Costs use the declared synthetic-API ledger and do not measure provider bills.

This is retrospective analysis of the tested candidate pools. It neither defines a new adaptive model selector nor demonstrates a prospective deployment improvement. E3 has six measured candidates, not complete physical outcomes for the 90 screened candidates. Repeats are not independent policy choices; campaigns are not pooled. E5's small cost gap is descriptive, not statistical evidence of a superior predictor.
