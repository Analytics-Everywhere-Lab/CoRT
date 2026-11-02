# Practical Numerical Replay

This public, CPU-only replay covers the three E4 sessions of the paper (Sec.
4.4 and 5.4): the earlier benchmark session, the removal-trained session and
the scene-only session. The [experiment guide](experiments.md#intervention-checks-e4-gemma-only)
maps each session to its job ID, aggregate record, config and runner; the job
IDs are the file names used below. The replay uses anonymous **unit sufficient
statistics**, not raw predictions, request logs, media, evaluator labels, or
model weights. It does not support independent per-request rescoring,
re-fitting, new model inference, or reconstruction of individual
candidate-order effects.

## Run

From the release checkout, with Python and NumPy installed:

```sh
python analysis/replay_practical.py
python analysis/replay_practical.py --output /tmp/practical_replay.json
python -m unittest discover -s tests -p test_practical_replay.py -v
```

The no-argument command deterministically writes
`results/rebuilt/practical_replay.json`. It runs without upstream data, network,
models, GPUs, or optional scientific Python libraries. Direct invocation also
works from another working directory. `--data-dir` selects a complete replay
bundle for integrity testing. It does not offer new seeds or draw counts.

The report has an overall `status`, per-session `status`, coverage, counts,
replayed results, decision-to-variant mappings, retention-reference mappings,
and additional timing-component means. A verification `pass` means the selected
historical statistics and decisions matched, **not** that the scientific gates
passed. Invalid inputs or mismatches raise an error and exit nonzero before
writing a new report; a pre-existing report is not overwritten on failure.

## Coverage

| Paper session (file) | Prediction variants | Source prediction rows | Source timing rows | Saved numeric references |
| --- | ---: | ---: | ---: | ---: |
| Removal-trained (`6469516.json`) | 26 | 23,244 | 16,640 | 1,345 |
| Scene-only (`6478505.json`) | 5 | 4,470 | 3,200 | 221 |
| Earlier benchmark (`621156.json`) | 18 | 16,092 | 11,520 | 3,116 |
| Total | 49 | 43,806 | 31,360 | 4,682 |

Each session evaluates the same reused 447 frames / 894 paired-order requests
across 50 units. These are not three independent participant samples. Each
session times 128 requests across 44 units with five repeats. The six untimed
units are absent from timing vectors, not zero-filled. Source strata are
`thu_read` (known participants), `escape`, and `youtube` (recordings with unknown
wearers). This is descriptive evaluation of reused development data, not
participant-generalization or fresh confirmatory evidence.

### Removal-Trained Session

Replays `counts`, `metrics`, `accuracy_differences_vs_native`, `timing`,
`overhead`, `decision`, and `per_source` numeric/boolean leaves:

- Unit accuracy and natural-log loss, both training seeds, both correction code
  paths (`legacy` and `fused` in variant names), zero-correction controls, and
  the ten within-unit seed means.
- Paired accuracy differences against the session's native baseline, and
  descriptive differences of the two co-primary policies from the native model
  with the post hoc answer-bias offset (`*_minus_native_frozen_bias`).
- Saved accuracy/loss estimates, two-sided percentile CIs, and one-sided family
  bounds; source-specific accuracy differences and their original intervals.
- Native/request latency means, ratios of pooled means, speedup CIs and family
  bounds, latency changes, forward means, memory means/maxima, and zero-control
  forward overheads. The replay additionally reports all four component means.
- Both co-primary decisions, which compare each policy with native. The
  policies are half scene and half candidates removed at L34 with readout
  correction, and all scene removed at L30 with `C union Q` correction. The
  config and report name them `E1` and `E2` (`decision.co_primaries`); these
  keys are unrelated to the paper's E1 and E2. **Both remain failures**: speed
  passes but accuracy does not.

The three answer-bias score vectors use the offset already fixed at
5.806 bits (`frozen_bias`). This replays its application to the
reused 894-request evaluation, not its fitting. The offset was fitted on 346
validation requests and evaluated on 244 test requests in an earlier
correction screen; those requests are not exported or relabeled as training
here.

### Scene-Only Session

Replays `counts`, `metrics`, `accuracy_differences_vs_native`, `timing`,
`overhead`, `decision`, `retention_diagnostics`, `per_source`, and
`per_source_retention` numeric/boolean leaves. This includes all five variants,
accuracy/loss CIs and family bounds, native-relative accuracy effects, speedup
CIs and bounds, latency and memory summaries, and both individual-seed gates.

Retention uses each policy's **matched unpruned readout adapter**, not the raw
native model. Accuracy and log-loss difference estimates/CIs are recomputed
both pooled and within source. These are descriptive differences, not
equivalence tests. **Seed 17 passes; seed 29 fails accuracy; `both_seeds_pass`
remains false.** The session's decision uses no seed averaging. As the paper
reports (Sec. 5.4), seed 17 was designated primary after both unpruned results
were known.

### Earlier Benchmark Session

Replays `counts`, `accuracy`, `accuracy_differences`, `timing`,
`diagnostic_decisions`, `primary_decisions`, and corresponding `per_source`
numeric/boolean leaves. This includes the 18 measured variants and five
accuracy/loss seed means, deletion-depth and correction-site comparisons,
paired native differences, CIs and family bounds, request/forward intervals,
ratios of pooled latencies, per-unit peak-memory intervals, observed memory
maxima, and preparation/lookup/forward/postprocess component means.

The paper's prespecified primary diagnostic (half the scene and candidate
positions removed at block 34) is recorded under `diagnostic_decisions`: its
compression accuracy and joint gates remain false, while its speed gate and
the correction accuracy diagnostic (`steering_accuracy_pass`) pass. The
separate `primary_decisions` block records `eligible: false` and
`confirmatory: false`: these are descriptive diagnostics, not confirmatory
successes.

This is complete replay of the selected numerical sections of this session's
saved summary, not the broader historical development grid. Earlier bias/tie
screens, answer-bias offset fitting, cold-start/setup costs, hardware
telemetry, projection receipts, and unselected technical metadata are outside
coverage.

## Statistical Contract

The exported `units` array follows the original analyzer's sorted unit order.
Only its positional index and allowlisted source/count fields survive; original
unit, frame, request, recording, and participant keys are not exported.
`timed_positions` selects the 44 measured units in their historical order.
Every variant shares that ordering. `timing_pairs` counts request-repeat pairs,
not independent sampling units.

Both candidate orders, all five timing repeats, and paired variants remain
within units. Seed means are calculated within the same units before
resampling. The exporter validates complete prediction inventories, paired
orders, paired timing request/repeat keys, and matched batch-one hardware.
The public arrays preserve the sufficient statistics for this unit bootstrap;
they do not expose the underlying request/order/repeat rows.

The replay uses NumPy's original default RNG, 10,000 draws, source order
`thu_read`, `escape`, `youtube`, and the original linear percentile quantiles.
Seeds are 20261019 for the removal-trained and scene-only sessions and
20261015 for the earlier benchmark session. Accuracy draws are consumed
**before** timing draws. Within-source runs restart the original seed
independently, as the original analyzers did. Family alpha is .05/4 for the
removal-trained and scene-only sessions and .05/3 for the earlier benchmark
session. Timing ratios use pooled means, not averages of per-unit ratios.
Absolute timing is never pooled across sessions.

The per-unit reducers also preserve the original distinction: NumPy means for
the removal-trained and scene-only sessions, `statistics.mean` for the earlier
benchmark session. Saved references come from original analysis summaries,
never from newly generated replay intervals. Numeric comparisons allow
absolute/relative floating-point tolerance 2e-12; boolean decisions must match
exactly. Runtime/library versions may affect bit-level output; matching is
checked rather than silently substituting intervals.

## Files and Curator Export

`results/replay_practical/` contains three `<session>.json` unit datasets, three
`<session>_expected.json` projections of saved scientific references, and
`manifest.json`. The manifest hashes only these public artifacts; it contains
no hashes or locators of private source records. The manifest detects accidental
changes; it is not a signature.

For an authorized curator with the original upstream inputs:

```sh
python analysis/export_practical_replay.py --upstream /path/to/research
```

`--output-dir` optionally selects a destination. The exporter reads upstream
without modifying it, verifies all 24 prediction/timing shard files against
their original worker-receipt hashes, validates the raw pairing contracts, and
replays all three projections against saved summaries before writing anything.
The upstream locators exist only in curator code, never in public numerical
artifacts. Original aggregate files and scientific analyzer source are unchanged.

Only source-stratum names, numeric unit arrays/counts, fixed variant names,
session names, original bootstrap settings, and projected scientific references
are published. Raw identifiers, filesystem paths, source media, source hash
lists, credentials, adapter manifests/weights, and training receipts are not
included in this numerical package. The separate adapter weights and provenance
inventory are documented in [adapters/README.md](../adapters/README.md).

## Verification

The focused tests compare every selected saved numeric/boolean field, exercise
the unchanged original scientific analyzers on entirely invented paired raw
rows, and independently verify RNG consumption order and matched retention
baselines. They also test failed scientific gates, missing timing membership,
duplicates/unpaired rows, schema/value/checksum tampering, privacy rules shared
with `analysis/export_aggregates.py`, deterministic CLI output, and aggregate
byte preservation. No private inputs are needed for these tests.
