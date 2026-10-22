# Package contents

`main.tex` and `supplement.tex` are separate anonymous submission entry points.
Build with pdfLaTeX and BibTeX. Bibliographies are generated from the included
`references.bib`; pre-generated `.bbl` files are not bundled.

See `README.md` for verification commands and separately distributed companion
materials, or `OVERLEAF_README_ZH.md` for Overleaf import instructions.
The prospective HTTP evidence and standard-library verifier are bundled in
`artifact/realapi_prospective/`; see its README for temporary-disk requirements.
The reduced AWQ predictions and Vidur control CSVs required by the bundled
verifiers are included. Full native GPU archives and model weights are separate.

The optional retrospective `artifact/14b_startup_sensitivity/recompute.py`
requires the frozen E6 package root in `FIDELITYLOOP_E6_RECEIVED`.
