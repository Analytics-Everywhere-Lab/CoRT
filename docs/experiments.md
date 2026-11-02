# Experiments and Protocol Status

This guide links each CoRT experiment to its paper results, plan, configs and
run records. The [paper-to-release map](../results/paper_results_map.csv) lists
the evidence file for every main table, numeric figure and supplementary table.

| Label | Paper results | Plan or config |
| --- | --- | --- |
| E0 | Sec. 5.1; Table S21 | `protocols/preregistration/behavioral_2afc_v1_2026_07_22.json` |
| E1 | Fig. 3; Table S12 | `protocols/preregistration/transport_decomposition_v1_2026_07_26.json` |
| E2 | Fig. 4; Table 2; Tables S13–S16, S22 | `protocols/preregistration/component_closure_v1_2026_07_26.json` |
| T1 | Table 3; Fig. 5; Tables S8–S9 | `configs/temporal/dense_current_v2.json`, `new_interaction_v1.json`, `interaction_order_v1.json` |
| E3 | Fig. 6; Tables 4–5; Tables S1–S7, S19 | Gemma: the E2 plan and `protocols/specifications/component_partition.py`; Qwen: `protocols/current/qwen_role_factorial_v1.json` |
| T2 | Fig. 7; Table S10 | `configs/temporal/new_interaction_followup_v1.json`, `interaction_order_followup_v1.json` |
| E4 | Table 6; Fig. 8; Tables S25–S36 | `configs/static_cort/recipes/` (see the session crosswalk below) |

## Claim status

The status words follow paper Fig. 2 and Sec. 4.3. Only Gemma E2 is
confirmatory: an internal test plan preceded its outcomes, without external
registration or certified timestamps, and the test confirms the direction of
the ordering at two preselected cuts. Qwen E2 is a planned descriptive check.
E0 is the eligibility rule and E1 is exploratory and hypothesis-forming. E3 is
descriptive, including the Möbius and Shapley analyses; the Gemma concentration
rule was prespecified, and the two Qwen contrasts are a planned follow-up on
the reused confirmation group, not independent confirmation. T1 and T2 are
exploratory and Gemma only. E4 is Gemma only and uses reused development data.

The directory name `protocols/preregistration/` is historical. Its files are
internal plans, not external registrations.

## Single-Frame Studies

| Stage | Design and status | Record |
| --- | --- | --- |
| Construction | Historical 800-frame pool; the plan's later sections were superseded and are not part of the paper | `protocols/historical_provenance/static_construction_retained_2026_07_21.json` |
| E0 | Behavioral group, 320 frames; A/B answer-likelihood eligibility test; prerequisite, not causal confirmation | `protocols/preregistration/behavioral_2afc_v1_2026_07_22.json` |
| Earlier audit | July 23 plan on the same 80 clusters as E1; an unreported audit that preceded the survey (Sec. 4.2) | `protocols/preregistration/answer_token_causal_intervention_v1_2026_07_23.json` |
| E1 | Exploratory group; four-cut, four-operator survey; exploratory, hypothesis-forming | `protocols/preregistration/transport_decomposition_v1_2026_07_26.json` |
| E2 | Confirmation group; Gemma L22/L34 reversal: internally planned confirmation; Qwen L24/L36: planned descriptive | `protocols/preregistration/component_closure_v1_2026_07_26.json` |
| E3 Gemma | All 15 C/Q/P/R unions at L34, Shapley, interactions, same-size benchmarks; descriptive, with a prespecified 50% concentration rule | Same July 26 plan; `protocols/specifications/component_partition.py` |
| E3 Qwen | All 15 unions at L36; two planned omission contrasts, other summaries descriptive | `protocols/current/qwen_role_factorial_v1.json` |

The July 21–24 pool, eligibility and earlier audits preceded the July 26 survey
and confirmation. The recorded chronology places E2's plan after the survey and
before E2 outcomes. The confirmation group was selected by excluding the 80
exploratory-group frames by participant, video, source frame and recorded
wrong-time frame. The recorded overlap audit also finds no shared participant,
video, frame or wrong-time frame with the behavioral group
(`results/aggregate/behavior.json`; Table S23). Its items appeared in earlier
pool and hidden-state capture runs. E3 Qwen reused those 46 clusters after the
Qwen cut-pair and Gemma decomposition results were known, and before its new
subset outcomes were inspected. Filename dates alone do not certify who saw
which outcomes.

The E1 operators are single scene replacement, persistent scene clamp,
complement replacement, and full-active replacement. Persistent-clamp cells
remain in the frozen outcome count. Their zero difference from single
replacement is structurally redundant under the matched construction, not an
estimated persistence effect or evidence about later recontamination.

Gemma E2 requires scene advantage at L22 and complement advantage at L34 each
above 0.10 recovery units: 10,000 paired participant-cluster bootstrap draws,
seed 20260727, Holm across two tests. Qwen E3 uses 20,000 draws, seed 20261014,
two Bonferroni one-sided 97.5% bounds, and a complete-K lower-bound guard above
0.10. Contrasts are `0.75*rho(K)-rho(QPR)>0` and
`rho(CQR)-0.75*rho(K)>0`. Export edits do not alter these rules.
Bootstrap seeds throughout this release are fixed identifiers, not dates.

## Clip Studies (Gemma Only)

| Stage | Task | Frames | Clips / participant clusters | Design | Config under `configs/temporal/` |
| --- | --- | ---: | --- | --- | --- |
| T1 | Current object (state) | 9 | 96 / 96 | B0, B1, B4, B8; H/V/K/A at L22/L34 | `dense_current_v2.json` |
| T1 | New interaction (event onset) | 17 | 37 / 20 | B0, B1, B4, B2; H/V/K/A at L22/L34 | `new_interaction_v1.json` |
| T1 | Interaction order (event order) | 17 | 27 / 14 | Same conditions, order-specific prompt | `interaction_order_v1.json` |
| T2 | New interaction | 17 | Same 37 / 20 | All 15 unions at L34, both directions and orders; S1, S2, RV | `new_interaction_followup_v1.json` |
| T2 | Interaction order | 17 | Same 27 / 14 | Same factorial and order controls | `interaction_order_followup_v1.json` |

Condition codes: B0 chronological clip; B1 final frame alone; B2 first frame
alone; B4 final frame repeated in every scene slot; B8 the eight earlier frames
shuffled with the final frame last; S1 and S2 two fixed shuffles of the 15
interior frames, first and final frames kept; RV reversed clip. Shuffled and
reversed inputs are scored for agreement with the original answer, not
accuracy.

All five studies are completed and exploratory. The current object config
(`dense_current_v2.json`) replaces v1's pinned record of previously exposed participants without changing the
scientific conditions. The clip sets were fixed before model outcomes. T1
analyses are described as fixed before outcomes were seen, without a dated
record. T2 and the 17-frame order controls were added after the first T1
results.

Labels were AI-assisted. One author reported validation of the final 37- and
27-clip sets after the first clip results, with no corrections. Independent
blinded annotation is not claimed. Shuffled and reversed inputs keep
original-answer agreement, not new gold labels. Later author validation does
not change the exploratory status or the previously exposed participant
population.

Local completion and verification records report 2,886 and 2,106 follow-up
cells and 7,770 and 5,670 passing checks. All 768 overlapping B0/B4/K/A
reference cells matched the original scores exactly. These are historical
verification results, not inference performed for this export.

Clip intervals use 19,999 paired participant-cluster draws: seed 20261023 for
current object and 20261025 for the two 17-frame tasks and T2. Orders are
averaged within clip and clips within participant. Intervals are unadjusted
descriptive 95% intervals, not cross-task, cross-cut or equivalence tests. CQR
excludes the entire prefix, including the readout.

## Intervention Checks (E4, Gemma Only)

The paper reports three E4 sessions on the same 447 reused development frames
(894 requests). The job IDs are internal provenance keys used in file names
under `results/replay_practical/` and in run records.

| Paper session | Job ID | Aggregate record | Config under `configs/static_cort/recipes/` | Runner |
| --- | --- | --- | --- | --- |
| Earlier benchmark session: deletion depth and frozen corrections | 621156 | `practical_compression.json`; also `bias_baseline.json` and `practical_followup_public.json` | `cross_source_validation_v1.json` | `decoder_cached_validation` (config freeze and summaries); its GPU worker `run_decoder_cached_validation` is not included |
| Removal-trained session | 6469516 | `practical_final.json`, key `combined` | `late_removal_training_v2.json`; executed: `late_removal_training_executed.json` | `run_role_aware_policy` (`train`, `freeze`, `evaluate`, `aggregate`); `role_aware_killarney_runtime` for `evaluate` and `aggregate` on Killarney |
| Scene-only session: native-trained readout adapters | 6478505 | `practical_final.json`, key `followup` | `late_scene_removal_eval_v1.json`; executed: `late_scene_removal_executed.json` | `run_role_aware_transfer` (`freeze`, `evaluate`, `aggregate`) |

Aggregate records are under `results/aggregate/`; runners are under
`research/lifecycle_memory/`. Two earlier development stages are not reported
in the paper's main text: a matrix of 71 reference and compression conditions
with 14 correction designs (seeds 17 and 29; `compression_and_correction_screen_v1.json`)
and early pilots (`early_application_pilots_v1.json`). The earlier benchmark
session evaluated eight reference or compression conditions and ten adapter
conditions selected from those results and fixed before that session.

Executed records are exact field projections of completed frozen configs, not
edits to pre-run plans. The scene-only design keeps its historical
`awaiting_freeze` status; the executed record has `adapters_frozen`.
Projection omissions and source/export hashes are explicit in the manifest.
Model, cut, seed, variant, tolerance and decision values remain unchanged.

In the earlier benchmark session, the prespecified primary diagnostic (deleting
half the scene and candidate positions at block 34) failed its −1 pp accuracy
margin. Deleting all scene positions after block 30 or 34 was cheap in
secondary settings without an equivalence margin. In the removal-trained
session, both co-primary seed-mean policies passed the speed check but failed
the accuracy check. Its config names them `E1` and `E2`
(`decision.co_primaries`); these keys are unrelated to the paper's E1 and E2.
In the scene-only session, seed 17 passed both checks and seed 29 failed
accuracy. Seed 17 was designated primary after both unpruned results were
known. Correction supplied the accuracy gain; removal reduced time at an
accuracy cost relative to correction alone.

Each later session allocates alpha 0.05 over four one-sided bounds (0.0125
each): 10,000 paired source-stratified sampling-unit draws, seed 20261019.
Passing requires an accuracy-difference lower bound above zero and a
native/policy speedup lower bound above one. Timing is warm, batch-one,
next-answer request time on four H100s, not long generation or throughput.
Latency changes compare within-session native baselines. No dated record
establishes that E1–E3 preceded these practical designs; the checks establish
agreement with the map, not that it guided selection.

## Inclusive Run Counts

Counts include technical full-active/self controls in their recorded inventories.
Separate technical check runs, retries, training steps and timing repetitions
are excluded unless identified. These are not independent sample sizes or
unique people.

| Completed inventory | Rows | Included detail |
| --- | ---: | --- |
| E0 behavioral group | 3,840 | 320 x 2 orders x 3 conditions x 2 models |
| E1 four-cut survey | 20,480 | 80 x 2 orders x 2 corruptions x 2 directions x 4 cuts x 4 operators x 2 models; includes 5,120 full-active and 5,120 persistent-clamp cells |
| E2 (both models) and Gemma E3 | 13,984 | Includes 1,472 full-active cells, not an extra 1,472 |
| Qwen E3 factorial | 6,256 | 46 x 2 orders x 17 position sets x 2 corruptions x 2 directions; includes scene and full-active |
| E4 earlier benchmark session | 16,092 predictions | 894 requests x 18 conditions |
| E4 removal-trained session | 23,244 predictions | 894 x 26 variants; 16,640 timing rows separately |
| E4 scene-only session | 4,470 predictions | 894 x 5 variants; 3,200 timing rows separately |
| T1 current object | 4,224 | 768 behavior + 3,072 patch + 384 self |
| T1 new interaction | 1,628 | 296 behavior + 1,184 patch + 148 self |
| T1 interaction order | 1,188 | 216 behavior + 864 patch + 108 self |
| T2 new interaction | 2,886 | 370 behavior + 2,368 patch + 148 self |
| T2 interaction order | 2,106 | 270 behavior + 1,728 patch + 108 self |

Listed inventories total 44,560 single-frame rows, 43,806 E4 predictions,
and 12,032 clip cells: **100,398 recorded rows**. This bookkeeping sum
includes reused data and controls, not study-wide compute. It excludes earlier
studies scored on generated JSON answers, earlier causal audits, the 71-condition development
matrix, pilots, training, separate technical check runs and timing rows. The
768 T1/T2 reference overlaps are rerun cells included in each run's inventory,
not new observations. Machine-readable counts and evidence hashes are in
`manifests/experiment_inventory.json`.

## Provenance and Replay

`manifests/source_to_export.json` binds plans/configs to repository-relative
sources and source/export byte hashes. Private account/container locations
are placeholders with explicit edit records; CPU/memory/topology and all
technical gates remain intact. The complete historical July 21 plan retains
superseded downstream sections and historical models without making them current
experiments. Source instructions describe historical study rules; this page
states current paper status.

Aggregate replay checks supported statistics from released sufficient statistics;
it neither reconstructs hidden request-level records nor supplies a new model
run. Embedded original hashes identify original artifacts, not sanitized bytes.

## Glossary of Internal Names

These names appear in module names, file names, record keys and seeds. They are
kept for provenance and carry no scientific meaning beyond what is listed.

| Name | Meaning |
| --- | --- |
| `lifecycle_memory` | Internal project name used as the Python namespace |
| `phase2`, `phase1`, `reanchored`, `patchaudit`, `job0`, `job3` | Internal project-stage labels in module and file names |
| Job A, Job D, Job E (`jobAR`, `jobD`, `jobE`) | Internal names for E0, E1, and E2 with the Gemma part of E3 |
| Killarney, Fir, TamIA | Slurm compute clusters where the GPU runs executed: E0–E3 and the earlier benchmark session on TamIA, the clip studies on Fir, the removal-trained and scene-only sessions on Killarney |
| `combined`, `followup` | Aggregate keys of the removal-trained and scene-only sessions |
| `transfer` (`role_aware_transfer`, `run_role_aware_transfer`) | Code name of the scene-only session; not a transfer test (the paper calls these development data, not held-out transfer) |
| cross-source (`cross_source_validation_v1.json`) | Code name of the earlier benchmark session, whose frames come from three sources |
| `heldout` (`run_patchaudit_heldout_component_closure`, `heldout_selection`) | The confirmation group; its items appeared in earlier pool and capture runs |
| `steering` (`application_pilot_steering`, `LowRankSteering`) | The learned activation correction |
| `temporal_forecasting` | Package for the clip studies. The paper reports only retrospective clip tasks; the package's `forecast_next_interaction` prompt is not used by any released config |
| dense current | Current object clips (T1, 9 frames) |
| recognition | The two 17-frame sets, new interaction and interaction order (T1) |
| follow-up | T2 and its order controls |
| `burned` | Items already exposed in earlier runs |
| canary | A short technical check run on a few requests before a full run |
| frozen, freeze | Fixed before the run it governs |
| `family_A` | The two-test post hoc sensitivity family (Table S24) |
| `scientific_supp_v2` | An archived earlier supplementary-material package used for construction records |
| schema-3, schema-4, schema-5 | Config format versions of the current object, 17-frame and T2 protocols |
