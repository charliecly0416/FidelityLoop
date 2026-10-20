# Final paper and artifact snapshot — 20 October 2026

This project snapshot includes the final paper and supplement, the full-day
arrival study, and both two/four-replica studies on the four-A800 host.
The paper tree matches the checked anonymous review snapshot. The project keeps
its installable framework, original paired corpus, and standalone CPU checks.

The added studies contain 8 full-day, 24 short-window capacity, and 12 pressure
executions. Their 44 executions remain separate from the original 197 physical
runs, 239 predictions and 555 pairs, and from the 12-run AWQ study. Failed deadline
outcomes remain included.

The protocol self-check now binds the current released paper. The previous
v8r13 binding is retained as
`artifact/bridge_execution/BASELINE_MANIFEST_v8r13.json`. The historical contract,
inputs and frozen implementation-footprint sources are unchanged. Protocol paths
are relative to the requested root, so relocated verification checks the copied
files. Regression tests cover relocated-input and paper-file changes.

Validation for this snapshot includes the project pytest suite, standalone
paired-corpus checks, E1 arithmetic, the frozen implementation-footprint audit,
and result checks for the added studies. See `REPRODUCING.md` and
`paper/README.md` for commands. The native CPU replay package is unchanged from
the verified 12-case, 25,200-decision snapshot.

Freeze tag: `project-freeze-20261020`.
