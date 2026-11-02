# Temporal Numerical Replay

This offline CPU replay reproduces the saved temporal participant-level estimates
and percentile intervals without models, media, evaluator labels, or network
access. It is a numerical analysis replay, not a reproduction of GPU inference
or dataset construction. All five studies (T1 and T2, Gemma only) remain
exploratory.

## Run

From the repository root, with NumPy installed:

```bash
python analysis/replay_temporal.py
python analysis/replay_temporal.py --output temporal_report.json
python -m unittest discover -s tests -p test_temporal_replay.py
```

The default output is `results/rebuilt/temporal_replay.json`. The report contains
`status` (`pass` or `fail`), `coverage`, `counts`, reconstructed statistics, and
separate comparisons against the exported source references and the unchanged
historical public aggregates. A numerical discrepancy produces a nonzero exit
code. Computation uses one numerical worker. Repeated execution with the same
inputs and NumPy version produces identical JSON; no timestamp or local paths
are recorded.

## Released Inputs

Files under `results/replay_temporal/` contain the following inventory. In the
file names, `recognition` marks the two 17-frame tasks and `followup` marks T2.

| Dataset | Study | Participants | Saved vectors | Vector entries | Recovery profiles |
| --- | --- | ---: | ---: | ---: | ---: |
| `current.json` | T1 current object | 96 | 306 | 29,376 | 16 |
| `recognition_new_interaction.json` | T1 new interaction | 20 | 309 | 6,180 | 16 |
| `recognition_interaction_order.json` | T1 interaction order | 14 | 309 | 4,326 | 16 |
| `followup_new_interaction.json` | T2 new interaction | 20 | 174 | 3,480 | 32 |
| `followup_interaction_order.json` | T2 interaction order | 14 | 174 | 2,436 | 32 |
| Total | | | 1,272 | 45,798 | 112 |

Each dataset has an explicit `anonymous_index_order` of integer ordinals. The
exporter validates the original sorted participant order before discarding the
identities. Every vector in a dataset uses this same order, retaining pairing
across conditions, directions, cuts, candidate-order strata, and role subsets.
Ordinals are local to a dataset and are not cross-study identity join keys.
The `path` arrays locate scientific statistics within the analysis structure;
they are not filesystem paths. Allowlisted condition labels retain their
scientific meaning. Saved numerical values are copied without rounding.

The T2 files also contain 64 anonymous clip-order records, each with the
five historical integer frame-order lists (`B0`, `B4`, `RV`, `S1`, `S2`; codes
in the [experiment guide](experiments.md#clip-studies-gemma-only)). These
are matched to the technical-check clip inventory before assigning sorted local
`clip_ordinal` values. The shuffles are not regenerated from anonymized names.
Clip ordinals are distinct from participant ordinals; no clip-to-participant
mapping or media lookup is provided. `B4` repeats the last frame, so its list is
not a permutation in the mathematical sense.

Technical summaries contain counts by check kind, covering 19,200 saved passing
checks. They summarize recorded flags, not a new execution of those checks.

## Estimators and Verification

Replay uses the released temporal analysis helpers, preserving the original
NumPy generator, bounded resampling batches, reduction order, and percentile
calculation. The current object seed is 20261023; the other four datasets
use 20261025. Every dataset uses 19,999 historical bootstrap draws. Candidate
orders and clips have already been averaged within each participant; participants
receive equal weight in the replay.

Recovery is the ratio of pooled participant means of residual reduction and
absolute donor discrepancy. It is not the mean of participant ratios. Numerator
and denominator share the resampled participant indices. The original native
gap and relative-denominator guards are retained: unstable ratios withhold
intervals without deleting participants or conditioning on surviving draws.

The replay checks 1,272 vector summaries and 232 ratio summaries, including all
112 top-level recovery profiles. The ratio count additionally includes 64
duplicated subset-profile summaries within the factorial sections and 56 derived
interaction, paired-difference, allocation, or relative-recovery summaries.
These are saved result locations, not independent experimental observations.

For each of four T2 direction blocks, all 15 nonempty role-subset vectors
support fresh computation of two paired contrasts, the C-Q recovery interaction,
and four Shapley allocations. This checks 28 derived vectors, their downstream
recovery and share intervals, and Shapley efficiency. The separately saved raw
gold-margin interaction vector supports its mean and CI replay, but does not
permit independent reconstruction of the per-example gold-oriented margins.

In the verified NumPy 2.4.6 environment, all 11,436 compared numeric values were
exactly equal, including 1,504 saved CI entries. The count includes derived
vector entries and numerical diagnostic fields, not just estimates. The report
separates exact equality from its absolute comparison tolerance of `2e-12`.
All exported intervals were available in these saved analyses; guard regression
tests also cover cases where an interval must be withheld.

Historical aggregate comparisons are intentionally narrower than source
comparisons. The public aggregates contain 696 of the 1,272 vector summaries;
576 candidate-order summaries are checked against their original source
references only. T2 public aggregates also omit 348 leave-one-participant-
out ranges, which are retained and replayed from the original sources. The 704
row-diagnostic fields attached to ratio records are copied and checked against
published values but are not recalculated from participant-level vectors.

## Offline Export

The public replay needs no upstream repository. Curators with the approved local
sources can regenerate the allowlisted projection explicitly:

```bash
python analysis/export_temporal_replay.py --upstream "$RESEARCH_REPOSITORY"
CORT_TEMPORAL_UPSTREAM="$RESEARCH_REPOSITORY" python -m unittest discover -s tests -p test_temporal_replay.py
```

The exporter reads five `analysis.json` files, five `technical_checks.json`
files, and two frozen permutation files using upstream-relative source
locations. It validates shared ordering, finite numerical values, scientific
labels, original/public reference agreement, and complete inventories before
writing. It verifies that source bytes have not changed. The optional source
audit checks every one of the 45,798 released vector entries against the original
sorted vectors and tests byte-for-byte repeatability of the complete export.

## Unavailable Inputs

This package does not publish source identities, media paths, provenance
hashlists, reviewer prose, raw result rows, evaluator joins, or answer mappings.
No models are run, and no downloads, pickle files, or evaluated code strings are
used. Historical aggregate files are read-only.

Independent rescoring of accuracy and gold-oriented margins still requires a
minimal authorized evaluator mapping and raw prediction inputs. Reconstruction
of participant means requires per-clip scores and their grouping. Actual runtime
reproduction additionally requires licensed media, frame manifests and temporal
alignment, model weights and processors, prepared requests, and intervention
execution. Raw technical certificates and frame-selection/QA evidence are not
replayed by these numerical vectors. No new GPU request examples or readiness
claims are supplied here.
