# Vidur request-output interoperability

Two CPU simulations use the public Meta-Llama-3-8B/A100/TP1 profile in
https://github.com/microsoft/vidur at commit
`abae7f63aa857300f5cdc6f5e0d27860cd24721b`.
Both use one simulated replica, Sarathi batch cap 32/chunk size 512 and seed 42.
The RF configuration is fixed at 100 trees, depth 8, min split 2, three folds
and two training threads. Lookup bounds are prefill 512, batch 32, tokens 1024.
These are bounded example settings, not the default Vidur accuracy benchmark.
The two synthetic inputs are fixed in `runtime/*/workload.csv`.

## Offline evaluation

From the Overleaf source root, with Python's standard library:

```text
python -B artifact/vidur_cpu/scripts/verify_adapter.py --case artifact/vidur_cpu/runtime/fixed32 --out artifact/vidur_cpu/analysis/fixed32
python -B artifact/vidur_cpu/scripts/verify_adapter.py --case artifact/vidur_cpu/runtime/mixed32 --out artifact/vidur_cpu/analysis/mixed32
python -B -m unittest discover -s artifact/vidur_cpu/scripts -p test_adapter.py -v
```

The first command reports (arrived, on-time, late, unfinished) counts of
(32,32,0,0), (32,32,0,0), (20,17,0,3); the second reports
(32,32,0,0), (32,16,16,0), (20,9,0,11).
The three protocol views are recomputations of each simulation, not separate
runs. All requests are online; the paper's P1+ criterion requiring 120 offline
requests does not apply. No lifecycle occupation or cost is inferred.
`analysis/` retains expected outputs. The evaluator is copied without changes;
`vendor/SOURCE.json` binds its content hash.

Native request CSVs and Chrome traces are retained under the successful
`native_output_dir` listed in each `run_metadata.json`. Chrome times use
microseconds; request records use seconds. Prefill produces the first output
token, so Chrome processing totals are input+output-1, while final request
progress is input+output. The CPU study independently checked all request
identities, token counts and completion times against both native formats.

## Optional native rerun

Download the pinned source archive from
https://codeload.github.com/microsoft/vidur/tar.gz/abae7f63aa857300f5cdc6f5e0d27860cd24721b
and verify SHA256
`a973e78e9a84ce0ab16c95de563c5024ba49e8559480375ee0319014af287058`.
Extract under `artifact/vidur_cpu/source/vidur-<commit>`.
Use an isolated Python environment and install `runtime/requirements-core.txt`.
The recorded execution used Python 3.13.5. Run the two cases in separate
processes, substituting that environment's Python below:

```text
python artifact/vidur_cpu/scripts/run_vidur.py --case fixed32
python artifact/vidur_cpu/scripts/run_vidur.py --case mixed32
```

No model weights or GPU are needed. First execution trains the configured
predictor; subsequent runs reuse its cache. The runner applies one export-only
fix, recorded in `runtime/native_export.patch`: a Python set becomes a sorted
list for JSON serialization. No prediction or scheduling logic is changed.
The first full simulation exposed this export error; both retained runs then
finished successfully. Native event_type=null is left unchanged; the included
Chrome/request formats provide the usable request evidence.

This bundle supports offline re-evaluation and a pinned-source rerun recipe.
It is an output adapter, not a full H/guard lifecycle integration or a physical
prediction-accuracy comparison. The source archive and trained cache are not
bundled. Upstream and vendored licenses are included. Internal absolute source
paths are removed from vendor provenance; scientific records are unchanged.
