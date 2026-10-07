# Paper release: 2026-10-07

Paper revision: v7r7. Baseline repository commit: 87deeb1.

## Scope and source review

The abstract explicitly introduces FidelityLoop as a framework presented and implemented in this work. Related work explains its policy-level validation role alongside performance simulators. The main text names the three existing rule baselines and points to their definitions and results in the supplement.

Vidur and Charon were checked against their official MLSys papers, including the abstracts and introduction/overview. The comparison describes their configuration-exploration role; it does not assert that either lacks a feature or that FidelityLoop outperforms them.

- Vidur: https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf
- Charon: https://proceedings.mlsys.org/paper_files/paper/2026/file/dbc8ce0fdfcd55172d73fb05dbae07fc-Paper-Conference.pdf

Evidence for the baseline summary is in `paper/external_validity.tex` and its generated W1/E1 tables. B-HPA is HPA-inspired, not a Kubernetes HPA reproduction; all three are implemented rule baselines, not full reproductions of the cited systems.

## Stage 1 checks

PASS: only `abstract.tex` and `body.tex` changed in the manuscript. Citation keys, labels, references, table inputs, and numerical evidence are preserved. No new empirical superiority or scalability claim is introduced.

## Build and delivery checks

PASS: main paper compiles to 10 pages and the supplement to 14 pages. The source ZIP was extracted into a clean directory and both entry points compiled successfully with pdfLaTeX/BibTeX via the LaTeX skill wrapper. Extracted PDF text matches the delivered PDFs, with no overfull boxes or unresolved citations/references. Existing template/font and underfull-box warnings remain.

The 10 main-paper pages were visually reviewed; the revised related-work and baseline passages were also inspected at full-page resolution. The supplement was unchanged in text and in all 14 rasterized pages, so its original PDF bytes are retained.

All three existing repository tests pass. The initial run detected an outdated public paper hash after recompilation; the public baseline manifest now binds the revised paper. Experimental protocol/input/result files remain unchanged. The release manifest excludes local pytest cache files that were previously listed despite not being tracked.

## Frozen deliverables

Tag: `paper-v7r7-20261007`. The software package remains version 0.1.0; this is a paper-only revision.

- `paper/`: anonymous source and checked-in PDFs.
- `FidelityLoop_Overleaf_v7r7_20261007.zip`: standalone source bundle attached to the GitHub release, with root-level `main.tex`, `supplement.tex`, and internal SHA256 checksums.
- `results/PAPER_VALIDATION.json`: machine-readable validation receipt and ZIP/PDF hashes.
- `artifact/MANIFEST.json`: current tracked-file inventory, excluding itself.

Import the ZIP as a new Overleaf project with pdfLaTeX and `main.tex`. This avoids overwriting manual edits in an existing project. Building `supplement.tex` produces the separate supplement. No direct Overleaf account changes were performed.

