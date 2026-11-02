# Temporal Replay Inputs

These five JSON datasets hold the T1 and T2 clip studies (Gemma only):
`current.json` is T1 current object, `recognition_*.json` are the two T1
17-frame tasks, and `followup_*.json` are T2. They contain 1,272 ordered
participant-level vectors with
45,798 unchanged numerical entries. They support offline replay of saved means,
percentile intervals, pooled recovery, and follow-up interactions and Shapley
allocations. The 112 top-level recovery profiles are included. Duplicated
factorial summaries are retained as separate historical result locations, not
additional observations.

Run from the repository root:

```bash
python analysis/replay_temporal.py
```

The deterministic report is `results/rebuilt/temporal_replay.json`; `--output`
selects another report path. The implementation uses one numerical worker and
the historical seeds and 19,999 draws. Its comparisons distinguish original
saved references from the narrower set present in existing public aggregates.

`anonymous_index_order` fixes the original sorted participant order before
identities are discarded. Arrays share that order within each dataset, keeping
numerators, denominators, conditions, and subset effects paired. Ordinals do not
provide cross-study identity joins. Recovery is a ratio of participant means,
not a mean of participant ratios.

T2 files additionally include frozen frame-order integer lists for 64
anonymous clips. Clip ordinals are distinct from participant ordinals; no media
or clip-to-participant lookup is included. Technical count summaries cover
19,200 saved passing check flags, without reexecuting the checks.

This is numerical replay, not full inference reproduction. Missing runtime
inputs include licensed media, frame manifests and alignment, prepared model
requests, model weights and processors, raw predictions, and an authorized
evaluator mapping. Row-level denominator diagnostics and selection/QA evidence
cannot be regenerated from these participant vectors. Source identities, media
locators, raw logs, provenance hashlists, and reviewer prose are excluded.

The [temporal replay guide](../../docs/temporal_replay.md) explains the schema,
offline curator export, exact comparison counts, and remaining reproduction
boundaries.
