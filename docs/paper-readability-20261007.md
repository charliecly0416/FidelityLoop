# V7r9 paper readability pass

Date: 2026-10-07

## Objective

Make the existing V7r8 manuscript easier to read on a first expert pass while
keeping its evidence, scope, and compact MLSys presentation unchanged. The pass
uses replacement and compression rather than new sections, experiments, figures,
or caveats.

## Stage 1: coordination and claim map

The required narrative is: FidelityLoop shares a policy contract between replay
and physical execution; replay trains and selects PPO checkpoints; frozen
checkpoints are then tested on the physical lifecycle; feasibility is checked
before cost. The guard remains a deadline-risk controller, not a correction to
the simulator. The edits preserve this distinction.

## Stage 2: execution

Applied changes:

- The abstract now states that PPO policies are trained in the framework replay
  environment, checkpoints are selected and frozen, and physical evaluation
  follows. It retains the original length and all reported numbers.
- The introduction uses the same train--freeze--deploy sequence and defines the
  all-on reference in plain language.
- The first problem statement identifies the delay-only synthetic API sink and
  scenario-USD cost; `all1` is identified as one GPU throughout.
- Workload classes receive a compact explanation of their online arrival shapes.
- The method distinguishes shared predictive/physical policy contracts from the
  separate CPU verifier that recomputes ledgers from raw records.
- The first model definition makes C's calibrated service times and original
  startup endpoint explicit. A redundant explanation of the deduplicated arm is
  shortened.

No table values, figure inputs, citations, labels, equations, experimental
protocols, or scientific claims were changed.

## Stage 3: review and verification

- Source anchor and math-block comparison: PASS; abstract and body retain all
  citation/ref/label/input anchors and equations.
- Main build: PASS, 10 pages, pdfLaTeX/BibTeX through the LaTeX skill wrapper.
- Supplement build: PASS, 14 pages; source and text are unchanged from V7r8.
- Visual inspection: main pages 1--10 and supplement pages 1--14 rendered with
  Poppler; no clipping, overlap, unreadable figure, or table defect was found.
- Repository tests: `python -m pytest -q`, 5 passed.

The paper scripts that parse only a single entrypoint report false missing
abstract/figure/table diagnostics because this project keeps those elements in
included files; the independent LaTeX build and rendered-PDF checks are the
authoritative checks for this layout.

## Freeze target

The resulting revision is V7r9. The previous V7r8 tag remains intact. The final
source ZIP is built separately and imported as a new Overleaf project; no live
Overleaf project is overwritten.
