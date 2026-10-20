# FidelityLoop v9r19 Overleaf package

Compile `main.tex` for the anonymous main paper and `supplement.tex` for the supplement.
The package includes all referenced PDF figures and pre-generated `main.bbl` and
`supplement.bbl`; Overleaf may regenerate them from `references.bib` if needed.
The `artifact/` tree contains the CPU verifiers and retained replay evidence.
For `artifact/14b_startup_sensitivity/recompute.py`, set
`FIDELITYLOOP_E6_RECEIVED` to the frozen E6 package root before running the
retrospective CPU verifier. No local absolute path is embedded in this package.
