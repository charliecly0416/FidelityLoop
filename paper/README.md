# FidelityLoop paper source and verification records

Compile `main.tex` and `supplement.tex` separately with pdfLaTeX and BibTeX.
All figures, table inputs, styles and bibliography files needed for compilation
are included; no shell escape or external downloads are required. The main
paper has a ten-page body, with references and AI Usage Disclosure afterward
(12 pages total in the checked build). The separate supplement has 36 pages.

## CPU checks

Run the following from the source root with Python's standard library:

```text
python -B artifact/input_cost_audit/verify.py
python -B artifact/awq_formal/verify.py
python -B artifact/vidur_control/verify.py
python -B artifact/vidur_service_screen/verify.py
python -B artifact/decision_price/verify.py
python -B artifact/decision_utility/verify.py
python -B artifact/e5_e6/verify.py
python -B artifact/e6_diagnostic/verify.py
python -B artifact/azure_diagnostic/verify.py
python -B artifact/capacity_transfer/verify.py
python -B artifact/capacity_pressure/verify.py
python -B artifact/realapi_prospective/verify.py
```

These checks recompute reduced-record results; they do not rerun GPU inference.
Each study directory describes its inputs, measures and evidence scope.
`artifact/vidur_cpu/README.md` also gives two offline request-output evaluations,
adapter tests and an optional pinned-source native rerun recipe.

## Companion materials

The anonymous release is available at
https://anonymous.4open.science/r/FidelityLoop-144B/.
The original paired-trace corpus, `cpu_replay/` and
`paper/artifact/full_day/` are distributed there separately from this source ZIP.
The full-day companion contains the recomputation tools and evidence for the
eight physical executions on two full-day arrival streams. Model weights and
complete native GPU journals are not included in this ZIP.

The original corpus contains 197 physical runs, 239 frozen predictions and
555 pairs, including all 41 strict deadline failures; prediction reuse does
not make the pairs independent trials. The AWQ context study adds 12 executions.
The later full-day, capacity-transfer and capacity-pressure studies add 8, 24
and 12 executions respectively. The prospective HTTP study adds 36 executions, with its sealed portable
evidence included in `artifact/realapi_prospective/evidence.zip`.
The supplement reports these evidence groups
separately. The four-replica pressure study includes service on both extra GPUs
after restart; the original short-window study has a different exposure scope.

## Source checksums

`PACKAGE_MANIFEST.json` lists source sizes and SHA256 hashes, excluding itself
and `SHA256SUMS.txt`. `SHA256SUMS.txt` covers all other source files and the
manifest. Build caches and compiled submission PDFs are not included.
`DISPLAY_NAMES.json` maps paper display names to archived IDs. Table regeneration
instructions remain in the corresponding artifact directories.
