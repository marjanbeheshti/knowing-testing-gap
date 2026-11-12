# Post-review analyses (exploratory)

Added after the protocol was frozen; reported as exploratory in the paper. Run from this folder;
results are read from `../../results`. Requires `numpy`, `scikit-learn` (and `evalplus` for `prompt_example.py`).

| Script | Paper |
|---|---|
| `paired_test.py` | Table II rows marked dagger (probe vs. prompt-only probe and output-level classifier); output-level vs. verbal confidence |
| `natural_green.py` | Table V(c) natural green set; likelihood ratios (RQ1); code + spec union (RQ1, G1) |
| `prompt_example.py` | RQ1 "Running the task's own example"; Fig. 1 candidate selection. Executes model-generated code; run in a sandbox or VM. |

`probe_v2.json` (per model, in `results/*_analysis/`) adds the logistic prompt-only probe; it is produced by
`src/probe.py` and reproduces the frozen `probe.json` rows exactly.
