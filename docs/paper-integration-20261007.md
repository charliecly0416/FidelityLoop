# Paper and reproduction integration, 2026-10-07

Baseline: paper v7r7, commit 44a2d0f. Target: v7r8, software 0.1.1.

## Stage 1: implementation and evidence

The historical facade validator now reads the identical simulator contract already distributed in `artifact/bridge_execution/inputs/`, and resolves its checkout root correctly. Its validation output must match `evidence/HISTORICAL_FACADE_VALIDATION.json` exactly.

`artifact/e1_repricing/` contains the previously verified fixed-event arithmetic implementation, tests, four reduced-data files, and reference outputs. Only packaging documentation, the license, and the verifier inventory change. No historical scientific input or pricing algorithm is revised. The E1 decision-replay summary and its independent audit are copied unchanged into `evidence/`; they document 16 runs and 29,580 ticks, not new experimental execution.

PASS: five repository tests pass, including exact comparison with the original facade receipt and relocation of the E1 package with its ten existing tests. The seven E1 analysis/test/reference/data files match the accepted source byte for byte. The public simulator contract is JSON-identical to the original input.

## Stage 2: manuscript and reproduction map

The method now connects the small synthetic interface tests to the existing 29,580-tick decision-reconstruction audit. The abstract distinguishes the study archive from the compact release. The supplement and REPRODUCING.md separate runnable portable checks from historical archive provenance. Citation keys, table inputs, and existing numerical evidence are unchanged; new numbers only expose the already accepted recovery result.

## Stage 3: clean builds and freeze

PASS: an independently copied source tree installed offline, imported its own package, and passed the smoke check, historical facade check, E1 verification, and all five repository tests. The facade receipt matches the historical receipt exactly. A separately extracted E1 ZIP passed its verifier, ten tests, and analysis; 169 numbers matched their references.

Both entry points compiled from the extracted Overleaf ZIP: main 10 pages, supplement 14 pages. Extracted PDF text matches the release PDFs, with no unresolved references/citations or overfull boxes. All 24 rendered pages were visually reviewed, with the changed method and reproduction passages inspected at full size. The reproduction paragraph was shortened to keep long commands in REPRODUCING.md and avoid broken command spacing. Existing template/font and underfull-box warnings remain.

The seven E1 scientific files and 48 existing figure/table/bibliography/protocol-input files remain byte-identical to their accepted sources. Citation keys and existing labels/references are preserved. The inherited recovery audit binds the summary for 16 runs / 29,580 ticks; this release verifies those receipts, not a new raw-GPU replay.

## Narrative closure

The paper retains the same chain: the framework connects replay and physical execution, the lifecycle diagnosis explains prediction mismatches, and rule/PPO execution tests feasibility before cost. The added method evidence verifies causal decision reconstruction within that chain; it is not a new contribution or a claim of prediction accuracy. No new experimental section, figure, table, or scientific outcome was introduced.

## Freeze

Freeze tag: `paper-v7r8-20261007`; software version: `0.1.1`. `results/PAPER_VALIDATION.json` binds the paper and ZIP checks. `artifact/MANIFEST.json` inventories the released source tree; release `SHA256SUMS.txt` binds the two ZIPs, PDFs, and receipts.
