# Public Numeric Results

The release covers every current experimental family: E0 behavioral
eligibility, the E1 depth survey, the E2 cut-pair test, the Gemma and Qwen E3
role decompositions, the T1 current object and 17-frame clips with the T2 role
decomposition, and the three E4 sessions (earlier benchmark, removal-trained
and scene-only). `coverage.json` and `paper_results_map.csv` identify the
evidence for each. Numbering follows the reviewed manuscript identified in
`paper_reference.json`.

## Paper-to-Release Map

`paper_results_map.csv` has one row per paper item or group of items:

| Column | Meaning |
| --- | --- |
| `paper_item` | Main or supplementary table, figure or section |
| `label` | The manuscript's LaTeX label; some labels keep internal names such as `jobe` or `killarney` |
| `family` | Result family; the same names group items in `coverage.json` |
| `evidence_files` | Released file(s) holding the numbers |
| `source_sha256` | SHA-256 of the original saved source summaries |
| `interval_status` | Status of the evidence file itself: `historical_finite_draw_replay_252_reference_values` (the four core replay datasets reproduce the saved intervals), `aggregate_only_not_exact_ci_replay` (exact projection of a saved summary; the file alone cannot replay intervals), or `derived_static_loco_not_in_252_reference_check` (leave-one-cluster-out rows regenerated from the replay vectors) |
| `note` | What the row covers |
| `additional_replay_report`, `additional_replay_guide` | The replay package report and guide that recompute selected statistics for the row, if any |
| `additional_replay_status` | `verified_selected_statistics` when that replay package passed in the bound receipt (`manifests/replay_verification.json`); `run_reproduce_all` when the receipt is missing or stale |

A row can therefore be `aggregate_only_not_exact_ci_replay` for its evidence
file and still have `verified_selected_statistics` from a replay package.
`source_receipts.json` keeps the family labels of the original curator export
(`practical_final_and_transfer`, `practical_earlier_transfer`); the map and
`coverage.json` use `practical_removal_trained_and_scene_only` and
`practical_earlier_benchmark`. These name the same E4 records; neither is a
transfer test.

## CPU Rebuild

From the release root, using Python 3.12 or newer and the pinned
`requirements.txt`:

```sh
python analysis/reproduce_all.py
python -m unittest discover -s tests -p test_results.py -v
```

Optional data-only PNG reconstructions of Figures 3-8 require Matplotlib and
Pillow:

```sh
python analysis/render_public_figures.py
```

This writes `figures/rendered/figure3.png` through `figure8.png`, using only
released numeric data. These are new plots, not copies of manuscript media.

The aggregate rebuild enforces one numeric thread and a 180-second deadline on POSIX.
It needs no network, model, source repository, or dataset. Repeated rebuilds are
byte-idempotent and never modify aggregate inputs, replay arrays or source receipts.
Outputs are numeric CSVs under `results/rebuilt/`, six figure-data CSVs under
`figures/`, the current coverage/map, and a verification report.

## Replay Boundary

The four dataset files in `replay/` (`gemma_cutpair.json`, `qwen_cutpair.json`,
`gemma_factorial.json`, `qwen_factorial.json`) contain the original anonymous
cluster sufficient statistics for E2 and E3. Their bytes and `expected.json`
are preserved, and `legacy_coverage.json` records their hashes, seeds, draws
and run inventories. They reconstruct 252 saved numeric reference values at
tolerance 2e-12, including historical finite-draw cut-pair/factorial
intervals. Anonymous ordinal indices preserve pairing and original resampling
order; they are not source participant identifiers.

```sh
python analysis/replay_cluster_sufficient_stats.py
python analysis/replay_cluster_sufficient_stats.py --seed 71 --output sensitivity_seed71.json
```

The second command saves independent sensitivity intervals, not historical
replay. Extra LOCO and raw-discrepancy summaries are derived from the same vectors
but are explicitly outside the 252-reference comparison.

The additional `replay_static_supplement/`, `replay_temporal/`, and
`replay_practical/` packages provide anonymous numerical inputs for the extended
replays. Their reports identify covered statistics and remaining exclusions.
The behavioral cluster vectors already in `aggregate/behavior.json` are also
used by the supplementary replay. See the package guides linked from the root
README for details.

Files in `aggregate/` remain exact projections of saved summaries, with
identifiers and private path metadata removed. They are independent references,
not overwritten by replay. `manifests/replay_verification.json` binds the extended
replay reports to their numerical inputs and code. The paper-results map links
these reports alongside the original aggregate evidence.
`source_receipts.json` records source SHA-256 values and selected JSON fields,
never original filesystem locations. Source-hash dictionaries are retained as
anonymous digest lists where their original keys contained private paths.

## Reading the Data

- `behavior.json`: `real` is accuracy; `blank` and `wrong` are real-minus-control
  accuracy gains, not absolute control accuracies. `pair` is both-orders correctness.
- `exploratory_depth.json`: `profiles` and `profile_cells` describe the 80
  exploratory-group clusters. The separate `gemma`, `qwen`, `primary`, and
  `sensitivity` sections concern the 46-cluster confirmation group. Do not pool
  these groups.
- `temporal_current.json` (current object) and `temporal_recognition.json` (the
  two 17-frame tasks): B0 chronological clip, B1 final frame alone, B2 first
  frame alone, B4 final frame repeated, B8 earlier frames shuffled with the
  final frame last. H, V, K and A are earlier frames, final frame, non-scene
  positions and all active positions. Both copying directions are retained.
- `temporal_followup.json` (T2): all 15 role subsets in both directions for each
  task, Shapley values, paired contrasts and order controls (S1 and S2 interior
  shuffles, RV reversal). Altered-order conditions are original-answer
  agreement, not accuracy. Figure 7 selects six B0-to-B4 L34 bars.
- `practical_final.json`: `combined` is the removal-trained session and
  `followup` is the scene-only session. Each uses its own native timing
  reference. Seed means are not individual policies or ensembles. Only seed 17
  of the scene-only session passes both fixed checks; seed 29 does not.
- `practical_compression.json`, `bias_baseline.json` and
  `practical_followup_public.json`: the earlier benchmark session. Despite its
  name, `practical_followup_public.json` belongs to the earlier benchmark
  session; the `followup` key in `practical_final.json` is the scene-only
  session.
- Missing timings remain blank in figure CSVs with `not_available_in_source`.
  Missing conditions are never invented or interpreted as zero.
- Numeric CSVs preserve Python round-trip representations and JSON pointers into
  their corresponding aggregate files. They are data tables, not manuscript facsimiles.

No source media, manuscript figures/PDFs, weights, checkpoints, activations, raw
trial/timing records, source identities, or private filesystem paths are included.
Conceptual Figures 1-2 and documentary tables are not numerical reconstructions.
No GPU experiment, retraining, or independent scientific replication is claimed.
