# Paper files

`main.tex` is the anonymous 10-page submission entry point. `supplement.tex` builds the independent 14-page supplement. All figures, generated table rows, style files, and bibliography required by these entry points are in this directory. `main.pdf` and `supplement.pdf` are the checked-in builds.

The paper reports a physical campaign on two A100 GPUs. The public repository provides the CPU framework and compact evidence ledger; it does not pretend that a CPU smoke test reproduces GPU timing.

## Overleaf

The source bundle for paper revision v7r8 is `FidelityLoop_Overleaf_v7r8_20261007.zip`, available from the GitHub release tagged `paper-v7r8-20261007`. Import the ZIP as a new Overleaf project, select `main.tex` as the main document, and use pdfLaTeX. Select `supplement.tex` to build the separate supplement. Both entry points are at the ZIP root; no external files or shell escape are required.

This revision integrates the existing decision-replay audit into the method and aligns the reproduction instructions with the portable CPU checks and E1 accounting artifact. Experimental evidence is unchanged. The source remains anonymous. Importing as a new project preserves any manual edits in an existing Overleaf project.
