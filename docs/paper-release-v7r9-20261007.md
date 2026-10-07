# FidelityLoop V7r9 paper release

Date: 2026-10-07

V7r9 is a paper-only clarity revision of V7r8. It makes the replay training,
checkpoint freezing, and physical policy evaluation sequence explicit and
separates predictive replay from the CPU accounting verifier. No experiment,
table value, figure source, citation, or software algorithm changed.

## Validation

- Main paper: 10 pages, independently compiled from a copied source tree.
- Supplement: 14 pages, independently compiled from a copied source tree.
- All citation, reference, label, input, and equation anchors are unchanged.
- Rendered main and supplement pages were inspected for clipping, overlap,
  figure clarity, and table legibility.
- `python -m pytest -q`: 5 passed.

Existing template/font and underfull-box warnings remain; there are no new
overfull boxes or unresolved references in the release builds. The parser-only
paper scripts do not follow this project's `\input` structure and therefore are
not used as the layout gate.

## Deliverables

The freeze tag is `paper-v7r9-20261007`. The source bundle is
`FidelityLoop_Overleaf_v7r9_20261007.zip`; it contains root-level `main.tex` and
`supplement.tex`, all figures and generated tables, and an internal SHA256 file.
The old V7r8 freeze and its assets remain available for comparison.

The source stays anonymous for review. Import the ZIP into a new Overleaf
project and select `main.tex`; build `supplement.tex` separately when needed.
