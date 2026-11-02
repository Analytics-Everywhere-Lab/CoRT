# Source Layout and Execution Boundaries

This is a **public source reference**, not a prepared GPU experiment release.
It preserves the `research.lifecycle_memory` namespace (an internal project
name), executable workers, analysis modules and local Python dependencies.
Running experiments also needs datasets, base models, containers, deployment
state and receipts that are not supplied. The 16 trained corrections are
provided in `adapters/weights/`. Successful imports or `--help` do not verify
the remaining inputs. Internal names are explained in the
[glossary](experiments.md#glossary-of-internal-names).

## Contents and Provenance

- `research/lifecycle_memory/`: E0 behavior, E1 depth survey, E2 cut-pair test
  and Gemma E3 decomposition, Qwen E3 factorial, and the E4 removal and
  correction source, plus their transitive imports and selected upstream test
  sources. It also includes the single-frame data builder, census/exclusion
  helpers, and the current object clip preparer with their local dependency
  closure; see `docs/preparation.md` for private input and SciPy requirements.
- `research/lifecycle_memory/temporal_forecasting/`: the complete clip-study
  package for T1 and T2 (current object, new interaction and interaction
  order, the T2 factorial and order controls), with data preparation, analysis
  and the original fail-closed launch contracts. Despite the package name, the
  paper reports only retrospective clip tasks. The package's
  `forecast_next_interaction` prompt is not used by any released config, and
  no forecasting results are released.
- `docs/research/...`: compatibility configs/profiles and the dynamic model
  helper at `docs/research/lifecycle_memory/phase 2 research/jobs/job0/run_phase1_qwen27b_hf.py`.
  These paths are part of the existing source interface, not deployment advice.
  The original private `heldout_selection.json` is deliberately excluded.
- `protocols/construction/`: documentary construction chronology and a separately
  named confirmation-group selection summary with identity arrays removed.
  Neither is an execution manifest; an independently obtained private execution
  manifest and its real hash-bound inputs remain required. See
  `docs/preparation.md`.
- `runtime/source_manifest.json`: relative upstream paths, original and exported
  SHA-256 values, byte counts and per-file transformations. It is distinct from
  the historical `source_manifest.json`, `release_manifest.json`, and temporal
  `release.json` formats consumed by experiment gates.
- `runtime/export_source.py`: offline exporter intended to run against the
  original upstream repository. It consults the existing Qwen, scene-only
  session and clip-study release selectors, then exports only the scientific
  import closure and compatibility configs, plus bounded archived construction
  projections. It does not run a model or prepare a deployment.

Scientific functions and gates are preserved. Four legacy helpers are reduced
to their imported symbols and transitive global definitions: contact-sheet
construction (`protocol_v3_3rscan_visual_vlm.py`), legacy scoring
(`build_epic_visor_dynamic_smoke.py`), parsing
(`run_epic_visor_dynamic_vlm_smoke.py`), and prompt helpers
(`run_epic_visor_dynamic_llamacpp_metal.py`). Their file names come from
earlier projects; no 3RScan, EPIC-VISOR or llama.cpp run contributes to the
paper. Retained statements are checked for AST equality before recorded comment
and docstring edits; unrelated
API/credential-handling commands are omitted. Machine-specific path/account
substitutions and editorial changes are recorded. Editorial replay rejects
changes to executable Python statements. Docstring edits also update CLI help
where it uses the module description. The SSH/deployment helper
`role_aware_killarney_transfer.py` is intentionally absent; it is not needed by
the scientific runners of the removal-trained or scene-only sessions.

## Actual Entrypoints

Run from the release root using `python -B -m research.lifecycle_memory.MODULE`.
Each module below provides a command-line interface. Append `--help` to inspect
its required inputs; the single-frame builder needs SciPy even for `--help`. The table is an
entrypoint map, not a claim that those inputs are distributed.

| Paper use | Lane | Module | Commands / execution contract |
| --- | --- | --- | --- |
| E0–E3 inputs | Single-frame data construction | `build_phase2_reanchored_egohos_battery` | Authorized inventories, images/masks, exclusion indexes and frozen prespecifications; NumPy, Pillow and SciPy required |
| T1 inputs | Current object clip preparation | `temporal_dense_current_data` | Private plan, media maps, single-frame indexes, per-clip review files and the record of previously exposed participants; no publication without them |
| E0 | Behavior | `run_phase2_reanchored_behavior` | `preflight`, `run-worker`, `technical-verify`; frozen public/model/job contracts and release bindings required |
| Hidden-state capture runs (not reported) | Depth capture | `run_phase2_reanchored_layer_capture` | `prepare-subset`, `run-model`, `aggregate`; prepared single-frame inputs and local model snapshots required |
| Earlier unreported audits (Sec. 4.2) | Causal patch / answer-token patch | `run_phase2_reanchored_causal_patch`, `run_phase2_reanchored_answer_token_causal_patch` | `preflight`, `run-worker`, `aggregate`; original patch contracts and technical gates apply |
| E1 | Four-cut depth survey | `run_patchaudit_transport_decomposition` | `preflight`, `run-worker`, `aggregate`; prepared inputs, prespecification and technical-check bindings required |
| E2 and Gemma E3 | Cut-pair test and decomposition | `run_patchaudit_heldout_component_closure` | `preflight`, `run-worker`, `aggregate`; frozen confirmation-group selection and public request pack required |
| Qwen E3 | Qwen factorial | `qwen_role_factorial` | `worker`, `aggregate`; prepared config, data directory, run directory, four ranks and a bound technical check run required |
| T1, T2 | Clip-study execution | `temporal_forecasting.cli` | `execute`, `worker`, `analyze`; verified clip-study release, deployment and dataset required |
| T1 current object | Current object launcher | `temporal_forecasting.dense_launch` | `check`, `canary`, `full` with a prepared deployment `--config`; submission additionally requires `--submit` |
| T1 17-frame tasks, T2 | 17-frame and T2 launcher | `temporal_forecasting.recognition_launch` | `check`, `canary`, `full`; the 17-frame (schema-4) or T2 (schema-5) protocol, its dataset bindings and a valid technical check run |
| E4 development (not in main text) | Development matrix and pilots | `run_decoder_token_matrix`, `run_decoder_application_pilots` | `--experiment` with prepared configs and data |
| E4 earlier benchmark session | Config freeze and summaries | `decoder_cached_validation` | `freeze`, `freeze-exposure`; its GPU worker `run_decoder_cached_validation` is not included |
| E4 removal-trained session | Adapter training and evaluation | `run_role_aware_policy` | `train`, `freeze`, `evaluate`, `aggregate` with `--config`; prepared data and adapters required |
| E4 removal-trained session | Evaluation on Killarney | `role_aware_killarney_runtime` | `runtime --stage evaluate\|aggregate --mode canary\|full`; original prepared `/mnt/rap` layout and frozen adapters required |
| E4 scene-only session | Freeze and evaluation | `run_role_aware_transfer` | `runtime --stage freeze\|evaluate\|aggregate --mode canary\|full --root PATH`; prepared frozen reference-adapter/data/config layout required |

`canary` is a short technical check run on a few requests before `full`.
`/mnt/rap` is the default root directory of the original prepared layout.

The matching `analyze_phase2_reanchored_*`,
`analyze_patchaudit_transport_decomposition`, and
`analyze_patchaudit_heldout_component_closure` modules analyze existing raw
outputs. Clip-study `analysis`, `dense_analysis`, `recognition_analysis`, and
`followup_analysis` are library modules dispatched by the clip-study CLI.
`temporal_forecasting.followup_compare` is a CLI for existing T2 results.
Aggregate publication tables are not substitutes for the complete raw inputs
these analyzers validate.

## Builders Are Not Standalone Deployments

The retained `*_release.py` helpers preserve upstream integrity contracts but
are **not turnkey commands against this public directory**. Their bundle,
preparation, verification and deployment workflows can require original job
scripts, Git-indexed inventories, previous releases, records of previously
exposed participants, models, adapters and hash-matched receipts. The manifest
lists excluded selector paths. The helpers fail if any input is missing or
does not match its recorded hash.

`temporal_forecasting.cli release` uses the upstream Git-indexed builder to seal
a supplied protocol, not this export's manifest. `temporal_forecasting.followup_prepare`
requires source deployment, results, verification, dataset, design and
human-validation artifacts. The Qwen config has unbound provenance/inventory
fields; the scene-only session config is `awaiting_freeze`. Both fail runtime
validation until prepared through their original workflows. Sanitized
path/account placeholders are not verified scheduler settings; model/data/runtime
pins remain unchanged.

## Runtime Pins and Missing Assets

The upstream GPU contract pins `torch==2.12.1+cu126` and
`transformers==5.13.0`. These are recorded in
`runtime/requirements-gpu-reference.txt`, not a complete installation lock.
The referenced container SHA-256 is
`6ee3e881a0fd3feaa6a5b19a5e0f3c104cf2250c3468f8d6254a04164adfb905`.
Model snapshots are `google/gemma-4-31B-it` at
`3548789868c5356dbf307c98e6f609007b82b3eb` and `Qwen/Qwen3.5-27B` at
`fc05daec18b0a78c049392ed2e771dde82bdf654`. No base-model weights or container are included.

Other execution paths import `qwen_vl_utils`, `torchvision`, `safetensors`,
`huggingface_hub`, `pandas`, `scikit-learn` or `matplotlib`; versions for these
are not established by this export. GPU execution requires the exact upstream
environment. Historical PyTest/Torch tests are outside the CPU-only suite.

Also absent: licensed RGB/video/crops, public request packages, private
evaluator labels, raw activation/score outputs, exposure and selection evidence,
training/evaluation/model inventories and training
receipts, validated scheduler accounts, Slurm allocation evidence and bound
technical-check/full receipts. Single-frame legacy job/prespecification
contracts not present at their original paths must be supplied by the original
preparation workflow. Use independently obtained, appropriately licensed,
correctly pinned assets; do not treat reporting manifests as prepared model
inputs.

## Offline Checks

```sh
python -B -m unittest discover -s tests -p test_source.py -v
python -B runtime/verify_source.py
python -B runtime/verify_source.py --upstream /path/to/original-repo --smoke
python -B runtime/cpu_checks.py --report runtime/cpu_test_report.json
```

The first two commands need only the standard library; the transport test skips
when Pillow is unavailable. `--smoke` and the broader suite are verified with
`runtime/requirements-cpu.txt`: NumPy 2.4.6 and Pillow 11.3.0 on Python 3.12.15,
matching the root CPU requirements. Model-framework imports are blocked. Smoke
reports the imported module count and loads job0 through its dynamic loader; the
two Torch-backed correction modules (`application_pilot_steering`,
`decoder_token_matrix_steering`) receive syntax/import-closure verification only.
Four single-frame preparation modules additionally require SciPy; if it is
unavailable, smoke explicitly reports them as source-only and optional
synthetic tests skip. Present but broken dependencies are errors, not silently
ignored imports.

The broader suite covers source integrity, partitions, E4 sidecar gates,
clip-study protocols, data validation and the T1/T2 analyses. Seven original
scene-only session tests (`test_role_aware_transfer`) expect adapters at
historical preparation paths and are explicitly skipped in the public CPU
runner; those paths are not populated by this export. The released copies in
`adapters/weights/` are checked separately against their recorded hashes. The
saved CPU report records counts and exclusions. CPU checks do not verify GPU
numerical equivalence, model inference, full dataset reproduction, latency,
cluster submission or published experiment outcomes.
