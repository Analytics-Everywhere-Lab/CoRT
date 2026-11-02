# Static Supplementary Replay

Run from the public checkout, offline, with Python and NumPy:

```sh
python -B analysis/replay_static_supplement.py
python -B analysis/replay_static_supplement.py --output results/rebuilt/static_supplement_replay.json
python -B -m unittest discover -s tests -p test_static_supplement.py
```

No arguments writes `results/rebuilt/static_supplement_replay.json`. `--check`
validates without writing. Exit code is zero only on success. The deterministic
report contains `status`, `counts`, and `coverage`; failure produces status
`fail`, empty counts/coverage, and an error type, never a partial success report.
The script forces numerical CPU thread counts to one and does not load models,
contact a network, read evaluator files, or alter existing aggregate/core replay
files. `analysis/reproduce_all.py` runs it with the other replay packages and
updates the release coverage record.

## Included

Parenthetical labels give the experiment and the paper table or section.

- Metric dependence (E3; Table S19): the archived anonymous CSV's 322 rows,
  seven strata, and 46 paired cluster ordinals. Both raw-margin and recovery
  interactions, covariance matrices, means, and intervals compare against saved
  references and the existing public aggregate. Regenerated 10,000-draw index
  arrays and paired interaction arrays must match SHA-256 digests of the
  original saved NPZ arrays, in little-endian int64/float64 C order. No NPZ is
  published.
- Behavioral replay (E0; Table S21): reuses the existing four 82-cluster vectors
  per model. Full intervals, one-sided bounds, centered add-one tails, and
  conjunction values compare to the original saved analysis, not regenerated
  references. Sampling uses the original `random.Random` streams, Python
  summation, and linear quantile expression. Blank/wrong vectors are paired
  differences.
- Exploration (E1; Fig. 3, Table S12): all 20,480 outcomes reduce to 128
  endpoint cells of 80 ordered cluster gap/error means. Includes all four
  sampled cuts, both models, corruption/direction strata,
  single/clamp/complement/full-sequence controls, the middle-cut four-test Holm
  family, and 64 plotted profile cells. Thick profiles average four
  stratum-specific recovery ratios; the middle-cut screen pools discrepancies
  first. These are intentionally different estimands. The .10 screen is a point
  cutoff, not the tail-test null.
- Gemma selected windows (E3 same-size benchmarks; Sec. 5.3): 4,416 outcomes
  reduce to 552 cluster/operator records across 12 controls. These are
  deterministic overlapping selected windows matched to Q/P/R cardinalities, not
  samples from a random reference distribution.
- Gemma stratification (E2; Table S16): 1,472 outcomes reduce to 736
  cluster/stratum/endpoint records across 16 cells. The late complete complement
  retains its original `CQPR` operator label; the early complement is `K`.
- Post hoc sensitivity (E2; Table S24): reuses existing Gemma cut-pair vectors
  for the two-test post hoc family (`family_A` in the records), with seed
  20260908 and 10,000 shared draws. It preserves threshold-adjusted discrepancy
  means, centered upper-tail ties, basic lower bounds, and Holm adjustment. Only
  this family is exported.
- Leave-one-cluster-out, LOCO (E2; Table S13): 46 rows recomputed from existing
  cut-pair vectors, preserving their ordinal order. Each omits one cluster and
  recomputes pooled recovery on the remaining 45. Historical checks cover saved
  minimum/median/maximum summaries, not saved per-omission rows. The full rows
  are explicitly regenerated output.

## Provenance And Limits

The exporter reads only from an explicit upstream root:

```sh
python -B analysis/export_static_supplement.py --upstream /absolute/path/to/research-checkout
```

It reconstructs the historical sorted analysis order while identities are in
memory, then exports numerical vectors with anonymous ordinal positions.
Ordinals are local to each dataset, not an identity crosswalk between cohorts.
The metric CSV is preserved byte-for-byte; JSON numbers are not rounded. NumPy
floating-point comparisons use relative/absolute tolerance 1e-12; metric draw
hash checks are byte-exact and can expose numerical-library differences.
Original shard/analysis/archive content digests bind provenance; no identity
hash lists, raw evaluator rows, media, source locators, or identity maps are
exported. File hashes also bind the unchanged public vector dependencies.

Only selected archive members are read. Obsolete sensitivity families, older
paper assets, and unrelated archive results are not published. The exporter
checks archived `family_A` vectors against the current public vectors without
duplicating them. Saved references and input hash validation are separate from
the replay computation. Export success includes running the public replay.

This is numerical replay, not fresh model evaluation, independent rescoring,
new inference, or a new interpretation of the science. Raw full-sequence maxima
are not reconstructed from cluster means; their endpoint recovery intervals are.
The 46 LOCO rows are regenerated diagnostics, with historical checks against
the reported minimum, median and maximum rather than per-omission references.
