# Data Preparation

This release includes the data-preparation implementations and their Python
dependencies. Prepared datasets and private execution manifests must be
supplied separately.
The offline exporter records original/exported hashes in
`runtime/source_manifest.json`. Executable scientific statements are preserved;
machine-path/account substitutions and comment or docstring edits are recorded.
No model execution is needed to
inspect the source or run the synthetic checks. Internal names are explained
in the [glossary](experiments.md#glossary-of-internal-names).

## Single-Frame Data

`research.lifecycle_memory.build_phase2_reanchored_egohos_battery` builds the
single-frame inputs used by E0–E3. It retains the component, crop, wrong-time
frame, allocation and exclusion logic. Its closure includes
`phase2_job3_egohos_census` (an annotation-only census of the EgoHOS frame
inventory and masks), `phase2_job3_egohos_burned_items` (items built from
previously exposed data), the marker renderer, and the two validators of the
recorded manual review of candidate items. The current upstream builder differs from
the older three-file construction snapshot in the archived `scientific_supp_v2`
package, which is not the paper's supplement. The
documentary crosswalk identifies that archived snapshot; the runtime source
manifest identifies the code distributed here.

Required external inputs include authorized EgoHOS images/masks, frame inventory,
candidate metadata, the participant mapping required by the census, the index
of previously exposed evaluator items, the governing and prompt-source
prespecifications, and any bound additional exclusion index. Review workflows
also require the original, consistent review records and receipts. Missing
inputs or mismatched pins remain failures. Use private output directories:
constructed crops, identities, evaluator joins and review records are not
automatically suitable for publication.

SciPy is required even to print `--help` for these two modules:

```sh
python -B -m research.lifecycle_memory.build_phase2_reanchored_egohos_battery --help
python -B -m research.lifecycle_memory.phase2_job3_egohos_census --help
```

Single-frame construction requires NumPy, Pillow and SciPy, including
`scipy.optimize.milp`. SciPy is not in the minimal CPU requirements, and no
tested full construction environment is claimed. Without SciPy the source can
still be read, `verify_source.py --smoke` reports the four affected modules as
source-only, and the single-frame synthetic tests explicitly skip. If SciPy is
present but broken or too old to import the required APIs, imports fail rather
than silently skipping. No optimizer is replaced and no frozen input gate is
relaxed.

## Current Object Clips

`research.lifecycle_memory.temporal_dense_current_data` is the bounded offline
preparer for the T1 current object clips (`dense current` in code). Its
dependency closure includes matched-media handling, matched preparation,
single-frame overlap and census helpers in addition to the clip-study package
already distributed. Inspect its actual CLI with:

```sh
python -B -m research.lifecycle_memory.temporal_dense_current_data --help
```

Inputs are a prepared plan/allocation directory (version 4), exact-URI local
media maps, authorized local clips, the original single-frame evaluator indexes
and image assets. Decoding needs local `ffmpeg` and `ffprobe`; review rendering
needs Pillow. No download, model or scheduler operation is performed by this
preparer. Full selection fixes 96 distinct allocated participants from metadata
before checking which clips are cached locally; the short technical check run
(`canary`) selects at most four cached participants. Original cohort allocation is
preserved. Nine decoded scenes are quality-check assets: only eight past
scenes enter the task, while the current frame and candidate crops keep their
original single-frame bytes and task binding.

Without a reviewer's per-clip review file, the output is only quality-check,
attrition and readiness reports, not a publishable dataset. Publication also
requires the current object protocol file (`schema_version` 3) and the pinned
records of previously exposed participants. Full publication requires all 96
fixed rows and remains descriptive. A caller's review file is an attestation,
not independent certification. Keep generated evidence private.

The following synthetic alignment example uses no media, labels or model and
does not construct a dataset or issue a review receipt:

```python
from research.lifecycle_memory import temporal_dense_current_data as dense

targets = [index / 4 for index in range(9)]
aligned = dense.choose_pts(list(range(9)), "1/4", 0, targets)
assert [aligned[slot]["pts"] for slot in dense.SCENES] == list(range(9))
assert dense._proofs(None) == {}  # No review file is invented when none is supplied.
```

`tests/test_source.py` also checks synthetic single-frame component
tie-breaking, candidate-only exclusions, missing/mismatched bound inputs,
full-selection failure for insufficient participants, and rejection of invalid
frame alignment. These checks do not establish end-to-end construction or
scientific equivalence.

## Documentary Records Are Not Runtime Inputs

`protocols/construction/` contains compact projections of the archived
supplementary-material package `scientific_supp_v2`, with original hashes and
recorded counts/times. No raw frame/request identity arrays, media locators or
reviewer-private records are included. The original runtime
`heldout_selection.json` (the confirmation-group selection) is excluded from
compatibility configs and is not published. Its original set/record hashes and
counts remain in `heldout_selection_documentary_summary.json`.

**An independently obtained private execution manifest is required** for the
confirmation-group runner and analyzer, along with the original bound request
pack and other original inputs. The documentary summary has a different
filename, schema and content hash. It cannot serve as the runtime pack, because it
lacks the membership arrays the runner checks. Historical hashes refer to
original records, not to redacted exports. As in the paper, the recorded
chronology is internal, not externally registered.

## Offline Export and Checks

```sh
python -B runtime/export_source.py --upstream /path/to/original-repo --destination /path/to/public-checkout
python -B runtime/verify_source.py --upstream /path/to/original-repo --smoke
python -B -m unittest discover -s tests -p test_source.py -v
python -B runtime/cpu_checks.py --report runtime/cpu_test_report.json
```

The exporter reads upstream only. It uses existing release selectors plus the
explicit preparation roots, excludes the original private selection pack, and
regenerates the documentary projections and manifest. It refuses an old public
checkout that still contains the private selection copy; removal must be an
explicit reviewed change. Upstream comparison checks source bytes after declared
sanitization and reconstructs the documentary projections from the archive.
