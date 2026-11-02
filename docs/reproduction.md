# Reproducing the numerical results

Run these commands from the repository root after installing
`requirements.txt` in a Python 3.12 or newer environment (tested on 3.12.15
and 3.13.2). The numerical replay uses a
CPU, does not download data, and does not load model weights.

## 1. Verify the release

```bash
python scripts/verify_release.py
python -B -m unittest discover -s tests -v
python -B runtime/cpu_checks.py
```

The release inventory records file sizes and SHA-256 hashes. The tests check
source dependencies, position partitions, protocol gates, result consistency,
and public-export hygiene. They are not a substitute for a short GPU technical
check run before new model execution.

The broader CPU suite includes synthetic execution and analysis tests. Eight
are skipped: seven scene-only session tests (`test_role_aware_transfer`) that
expect adapters at historical preparation paths, and one single-frame
preparation test that needs SciPy. The released adapter copies are verified
separately by the release tests.

## 2. Replay the numerical analyses

```bash
python analysis/reproduce_all.py
```

This runs the core static (E2, E3), supplementary static (E0, E1 and
single-frame checks), temporal (T1, T2), and practical (E4) replays in
sequence. It then checks that inputs were unchanged and writes
`manifests/replay_verification.json`, binding the reports to their input and
implementation hashes. Add `--figures` to also render the data-only plots.
Each subprocess uses one numerical thread and a 15-minute safety timeout.

The packages can also be run separately:

```bash
python analysis/replay_static_supplement.py
python analysis/replay_temporal.py
python analysis/replay_practical.py
```

See the [static supplementary](static_supplement_replay.md),
[temporal](temporal_replay.md), and [practical](practical_replay.md) guides for
their exact field coverage, resampling contracts, and exclusions. A replay
`pass` means the saved result was reproduced; it does not turn a failed
experimental test into a scientific success.

The original static replay of four datasets (Gemma and Qwen E2 cut-pair, Gemma
and Qwen E3 factorial) remains available independently:

```bash
python analysis/replay_cluster_sufficient_stats.py
```

This recomputes the E2 cut-pair comparisons and E3 role decompositions from
anonymous cluster sufficient statistics. It preserves the
original cluster order, random seeds, and bootstrap draws. The expected result
is four datasets with no discrepancies across 252 saved numeric reference
values, using an absolute comparison tolerance of `2e-12`.

The replay checks include recovery, paired contrasts, Shapley allocation,
interaction terms, and the supported statistical tests. It does not re-run
inference. An optional `--seed` evaluates sensitivity to a different bootstrap
seed; that output is deliberately not labeled historical replay.

## 3. Rebuild result tables and figure inputs

```bash
python analysis/rebuild_public_results.py
python analysis/render_public_figures.py
```

Outputs include:

- `results/rebuilt/`: replay results, supplementary static diagnostics,
  numeric exports, and a verification summary;
- `results/paper_results_map.csv`: paper tables and figures linked to their
  released evidence files and replay status (columns explained in
  `results/README.md`);
- `figures/figure3_data.csv` through `figures/figure8_data.csv`: plot inputs;
- `figures/rendered/`: data-only visualizations, without source images;
- `results/coverage.json`: coverage and reproduction limits by experiment.

The `aggregate/` files retain the recorded estimates and intervals. Additional
anonymous input packages support the temporal, practical, and supplementary
statistics listed in their guides. The build checks the replay receipt before
marking those packages verified; changed inputs or code invalidate that status.
Per-request outputs and private identity mappings are not included. Additional
static diagnostics remain separate from the original 252-value comparison.

## 4. Inspect or extend the experiments

Start with the [experiment guide](experiments.md),
[environment record](environment.md), and [data-access guide](data_access.md).
The [preparation reference](preparation.md) explains the single-frame and
current object clip builders and the authorized inputs they require.
The implementation retains its original `research.lifecycle_memory` namespace
(an internal project name) and module names for comparison with the recorded
experiments.

New model runs require separately obtained media, annotations, base-model checkpoints,
prepared input manifests, and a compatible GPU runtime. The public source
export has its own hashes; original execution receipts identify the source and
inputs used for the reported runs. Rebuilding from this export creates a new
run, not a replacement for those receipts.

## Interpreting the outputs

Recovery measures movement toward a donor's answer margin under a specified
replacement. It is distinct from answer accuracy. A joint candidate-and-question
patch is also distinct from the sum of their Shapley allocations.

The practical timing sessions retain separate native baselines. Compare a
condition with the native baseline in its own session, rather than subtracting
latencies measured in different sessions. See [method summary](method.md) for
the role definitions and [experiment guide](experiments.md) for result status.
