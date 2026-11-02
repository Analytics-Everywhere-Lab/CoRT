# Data Access and Reconstruction

This release contains CoRT code, frozen specifications, selected derived
numerical records, and the [trained adapter weights](../adapters/README.md).
It excludes source media, raw source/participant identifiers,
per-example evaluator mappings, and base-model weights. Apache-2.0
applies to the released CoRT code and adapter weights. Dataset, annotation, model, tokenizer, and
processor licenses remain independent; the code license does not relicense them.
These exclusions define this release, not a blanket legal conclusion that all
derived records are prohibited from redistribution.

## Provider Access

| Material | Provider documentation | Use in CoRT |
| --- | --- | --- |
| Ego4D recordings and metadata | [Access](https://ego4d-data.org/docs/start-here/), [metadata schema](https://ego4d-data.org/docs/data/metadata/) | Single-frame media lineage, participant grouping, clip video |
| EgoHOS imagery and role masks | [Repository](https://github.com/owenzlz/EgoHOS), [download script](https://github.com/owenzlz/EgoHOS/blob/main/download_datasets.sh) | Single-frame role annotations and the E4 frames |
| Ego4D short-term anticipation annotations | [Benchmark](https://ego4d-data.org/docs/benchmarks/forecasting/), [annotation schemas](https://ego4d-data.org/docs/data/annotations-schemas/) | Event proposals for the 17-frame clips, not automatic direct-manipulation labels |

Obtain access directly from the providers under their applicable terms.
EgoHOS contains several source lineages. Its Ego4D-derived imagery is distinct
from its THU-READ, escape-room, and YouTube imagery; origin terms remain relevant.

## Dataset Lineage

| Group | Frames or clips | Statistical groups | Relationship |
| --- | ---: | ---: | --- |
| Original single-frame pool | 800 frames | 147 participants; 313 videos | Previously examined pool |
| Behavioral group (E0) | 320 frames | 82 participants; 174 videos | Source/wrong-time-frame exclusions from the pool; participant/video reuse permitted |
| Exploratory group (E1) | 80 frames | 80 participants | Outcome-blind subset of the behavioral group |
| Hidden-state capture subset | 96 frames | 96 participants | From the pool; parent of the confirmation group |
| Confirmation group (E2, E3) | 46 frames | 46 participants | Selected by excluding the exploratory group; no overlap with the behavioral group either (Table S23); previously exposed in pool and capture runs |
| Adapter training | 505 frames | 88 participants | Separate Ego4D frames from a participant-based training split (not the provider's split); 1,010 paired-order requests |
| E4 evaluation | 447 frames | 50 sampling units | 83 THU-READ, 106 escape-room, 258 YouTube frames; reused across sessions |
| T1 current object | 96 clips | 96 participants | Previously exposed participants |
| T1/T2 new interaction | 37 clips | 20 participants | Same clips reused for T2 |
| T1/T2 interaction order | 27 clips | 14 participants | Same clips reused for T2 |

E4 units are eight known THU-READ participants, 15 escape-room recordings,
and 27 YouTube recordings, not 50 identified people. The two 17-frame sets
share 13 participants and 20 source-event proposals, with task-specific
windows: their 64 clips represent 21 distinct participants, not 34 independent
people. These groups cannot be summed to obtain a unique study-wide participant
or frame count.

## Single-Frame Metadata and Rendering Recipe

Reconstruction joins the EgoHOS image/mask inventory and Ego4D source mapping
to official Ego4D v2 metadata (`ego4d.json`), using `video_uid` to resolve
`fb_participant_id`. Missing/conflicting provenance does not establish an
independent person. Plans retain original sidecar/exclusion hashes; corresponding
identity-bearing records are not shipped. The July 21 historical specification,
July 22 behavioral specification, and July 26 component specification distinguish
the pool, E0, and E2 selection rules. Source code references and hashes are in
`manifests/documentation_provenance.json`.

1. Frames need usable first-order classes 3/4/5 and second-order classes 6/7/8.
   Classes 3/6 require left-hand pixels, 4/7 right-hand pixels, and 5/8 both.
   These are annotation-native roles and hand-presence checks, not an independent
   physical-contact or object-interaction-chain audit.
2. Each class uses its largest 8-connected component of at least 64 pixels;
   equal-area ties use lexicographic top/left/bottom/right bounding boxes.
   One square 3-by-3 erosion with zero exterior must leave at least 64 pixels.
   A failing largest component does not fall back to a smaller one.
3. Hash-ordered allocation selects one first/second-order pair per frame.
   The original pool fixes 800 frames, at most ten per participant, and at least
   100 participants. Class and pair balance are soft optimization objectives.
   The capture subset takes 96 hash-ranked participants and one hash-ranked
   frame each. E2 retains eligible capture pairs after excluding E1 by
   participant, video, source frame, and recorded wrong-time frame.
4. Crops retain component-interior RGB on black, with four native pixels of
   bounding-box margin, clipped to source bounds. Lanczos fit-and-center uses
   black 336-by-336 canvases. The EXIF-normalized unmarked RGB scene uses a
   black 896-by-672 canvas. Both A/B crop orders are retained.
5. `blank_full` is RGB (0, 0, 0). `wrong_time` is the first hash-ranked eligible
   different frame from the same video with source-frame-index gap at least
   900. This is not a seconds threshold, nearest-time match, or answer-preservation
   audit. Both corruptions retain original labels and identical candidate crops.

The original allocation uses SciPy MILP with zero relative MIP gap, 120-second
time limit, and presolve enabled; nonoptimal termination fails construction.
Objective weights are 1,000,000 for class deviations, 100 for pair deviations,
-10 for participant coverage, and 0.001 times the normalized hash rank for
tie-breaking. Selection uses the hash namespace
`phase2_reanchored_egohos_2afc_selection_v1`; capture uses
`phase2_reanchored_egohos_mechanism_subset_v1`. These strings are fixed salts
whose words are internal project labels. They are existing source settings,
not a new selection algorithm. Exact cohort identity still depends on the
original input inventory and exclusions.

E4 evaluation keeps the task, rendering, and prompt, using real scenes
only. Historical exclusions and duplicate checks precede scoring. Sources and
within-source units receive equal weight; requests and seeds are not independent
units. Visual QA covered 36 overview and nine larger examples, not exhaustive
human reannotation.

## Clip Metadata and Review

Current object clips (`dense current` in code) keep the original single-frame
scene as the final frame and the original candidate crops, adding eight earlier frames at -2.00
through -0.25 seconds in 0.25-second steps. Exported-clip boundaries,
source-frame timing, and decoded presentation timestamps establish alignment.
Decoded final frames are alignment references, not replacements for the
single-frame image. All 864 sampled timestamps were at or before their targets,
with maximum lag 33.3 ms. Screening of 1,120 historical rows left
775 eligible rows from 123 participants; outcome-blind hash selection chose 96
distinct participants without cache preference. Original split names remain
provenance, not confirmation status.

The 17-frame clip proposals (`recognition` in code) use official anticipation
frame/time-to-contact metadata, two different-category events 1–6.5 seconds
apart, and 0.75-second padding. There were 177 proposed event pairs across 37
participants and 174 available new-interaction subwindows. Review used actual
frames/crops in round-robin participant order; selected sets retain at most
three clips per participant. Windows are at most eight seconds, with 17 evenly
spaced target times. Each sampled frame is the nearest decoded frame at or
before its target within 0.1 seconds; timestamps must strictly increase.

Unclear onsets, tool-only contact, unmatched instances, indistinguishable crops,
ambiguous order, and new-interaction windows where both candidates start were
excluded before original model execution. Labels were AI-assisted. One author
reported validation of the final 37/27-clip sets after the first results, with
no corrections. Exact review time, blinding, and inter-rater agreement were not
supplied. This attestation does not cover all proposals, current object clips,
or shuffled/reversed clips.

T2 reuses the same clip sets, tests all 15 nonempty C/Q/P/R unions, and adds two
distinct endpoint-preserving interior shuffles and exact reversal. Transformed
inputs retain the original prompt and answer for agreement scoring; they do not
have newly validated event labels. Exact task prompts are in `configs/temporal/`.

## Reconstruction Boundary

Exact request-level reconstruction also requires identity-bearing selections,
exclusions, evaluator mappings, authorized media, and pinned model/processor
snapshots. Recipes and counts alone cannot recreate the exact cohorts.
Model-facing requests and technical check runs remain evaluator-unjoined; labels
are joined only after identity, completeness, geometry, and numerical gates.
Media, raw IDs, evaluator rows, and base-model weights are not included in this release.
