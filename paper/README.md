# FidelityLoop paper and result checks

[Main paper](main.pdf) · [Supplement](supplement.pdf) ·
[Original paired corpus](../README.md) · [Native CPU replay](../cpu_replay/README.md)

The main PDF is the checked Overleaf download for this review snapshot. It has
a ten-page body, followed by references and AI Usage Disclosure. The supplement
contains the experimental contracts and extended results, including the full-day
and same-host capacity studies. Compile `main.tex` or `supplement.tex` with
pdfLaTeX and BibTeX; figures and the MLSys style files are included.

## Recompute the added experiments

Run from this `paper/` directory with Python 3.10 or newer:

```text
python -B artifact/full_day/tools/recompute.py
python -B -m unittest discover -s artifact/full_day/tests -v
python -B artifact/capacity_transfer/verify.py
python -B artifact/capacity_pressure/verify.py
```

The [full-day audit](artifact/full_day/README.md) contains all eight executions
and 24 primary predictions. The [capacity audit](artifact/capacity_transfer/README.md)
contains 24 executions and 48 predictions on the original short windows. The
[pressure audit](artifact/capacity_pressure/README.md) contains 12 executions
and 24 predictions on the constructed offline-pressure input. Their 44 physical
executions are separate from the historical 197/239/555 corpus and from the
12-run AWQ study. Every deadline failure remains included.

The capacity checkers recompute per-request deadlines, occupied resources,
scenario costs, device service exposure, and frozen candidate selection. The
original-window study requires feasibility on two windows; the pressure study
uses one window. Both retain cost ties and report repeat 1 as primary and
repeat 2 as robustness. The full-day checker retains its day/host pairing.

## Other result checks

Run these from `paper/`; each uses the standard library unless its README says
otherwise:

| Command or directory | Checked evidence |
|---|---|
| `python -B artifact/awq_formal/verify.py` | 18 frozen predictions and 12 AWQ executions; eight policy runs enter the three-predictor comparison, with both failures retained |
| `python -B artifact/input_cost_audit/verify.py` | 28 online source-row intersections, three hash bindings, fixed-policy cost means, and the 16-ratio observed-repeat envelope |
| `python -B artifact/vidur_control/verify.py` | 12 retrospective predictions, deadline/resource errors, and target agreement against 36 physical runs |
| `python -B artifact/vidur_service_screen/verify.py` | Five service-screen metrics from 90 request records, separate from the policy-run corpus |
| `artifact/vidur_cpu/README.md` | Native-output evaluation, nine adapter tests, and the pinned Vidur rerun recipe |
| `python -B artifact/decision_price/verify.py` | Nine GPU/API price combinations and their deployment choices |
| `python -B artifact/decision_utility/verify.py` | Deployment choices and observed consequences in E2–E6 candidate pools |
| `python -B artifact/e5_e6/verify.py` | Original E5/E6 comparison and startup-only contrast |
| `python -B artifact/e6_diagnostic/verify.py` | Retrospective 30-second startup contrast and event chain |
| `python -B artifact/azure_diagnostic/verify.py` | Retrospective Azure comparison under strict request deadlines |
| `artifact/e4_startup_sensitivity/recompute.py` | Engine-study startup sensitivity |

The input-overlap checker establishes source-row separation, not session or
prompt independence. Reduced records support arithmetic and consistency checks;
they do not rerun GPU inference or authenticate the private machine journals.
The 30-second, Azure, and Vidur analyses are retrospective and do not increase
the original frozen-prediction counts.

The optional `../cpu_replay/` runner uses bundled source, measured profiles,
and pinned dependencies to regenerate the 12 fixed Vidur predictions.
Model weights and full GPU archives are not included. `DISPLAY_NAMES.json`
maps display names to unchanged archive IDs. If regenerating the original
tables, run `python refresh_display_tables.py` afterward to update display names.

`SHA256SUMS.txt` binds this paper tree. The root manifest binds the complete
snapshot, while `INDEX.json` defines the original corpus. The review address remains
https://anonymous.4open.science/r/FidelityLoop-144B/.
