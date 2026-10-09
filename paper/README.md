# Paper

`main.tex` builds the anonymous paper with pdfLaTeX and BibTeX; `supplement.tex` builds the independent supplement. All required source, figures, tables, and style files are included. No external files or shell escape are required.

Import this complete ZIP into Overleaf and select the desired entry point. The current build has 11 main PDF pages (body through page 10; references continue on page 11) and a 18-page supplement. `SHA256SUMS.txt` covers the source files.

This bundle preserves the experimental evidence, clarifies the deployment conditions behind the cooldown diagnostic, and summarizes practical validation steps. It documents the paired-trace release: 132 physical runs, 140 unique predictions, and request-level analysis tools. The trace archive is distributed separately from the Overleaf sources. E3 finds three false rejections by C among six physically feasible new checkpoints; it does not establish a generally superior screening model. The implementation-footprint audit separates policy code, integration code, and reused dependencies; the bibliography uses verified conference publications. Checkpoint provenance and the unified cost metric remain documented in `PPO_TRAINING_CONTRACT.json` and `COMMON_COST_METRIC.json`.

The 2026 MLSys CFP allowed 10 body pages excluding references. The 2027 CFP was not yet posted when checked on 2026-10-08; verify the target-year rules and template before submission.

The paired trace corpus is available for anonymous review at https://anonymous.4open.science/r/FidelityLoop-144B/ . This source bundle adds only that access link to the frozen paper.
