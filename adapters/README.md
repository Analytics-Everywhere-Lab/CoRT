# Trained activation corrections

`weights/` contains the 16 original adapter checkpoints evaluated in the three
E4 sessions of the paper (Sec. 4.4 and 5.4), totaling 2,759,024 bytes. Adapters
for the other designs of the earlier development matrix are not included. `manifest.json` records their SHA-256 hashes,
sizes, tensor shapes, targets, training seeds, and fixed final training step.
Duplicate copies used in multiple sessions appear only once. These files match
the original checkpoints byte for byte; they have not been converted or retrained.

Each adapter contains two FP32 matrices with 43,008 parameters and rank four.
These are activation corrections applied after a decoder block, not LoRA updates
to the base model's weights. The base model, optimizer states, and training media
are not included. Numerical replay of the published results does not require
loading the adapters.

The checkpoint names retain the scientific configuration keys used by the
experiment code. `native` means training without token removal; the other prefixes
identify the removal or merging configuration. `CQ`, `FQ`, and `readout` identify
candidate-and-question, scene-and-question, and answer-readout correction targets.
Each configuration has separate seed-17 and seed-29 checkpoints, selected at the
fixed final step 256. They are individual adapters, not an ensemble.

| Checkpoint prefix | Token configuration during training | Correction target after L34 | E4 sessions and paper use |
| --- | --- | --- | --- |
| `native__correct_L34_CQ` | No removal or merging | Candidate images and question | Earlier benchmark (Tables S30–S32) and removal-trained reference; compared with the readout target without removal (Sec. 5.4) |
| `native__correct_L34_FQ` | No removal or merging | Scene and question | Earlier benchmark only (Tables S30–S32) |
| `native__correct_L34_readout` | No removal or merging | Answer-readout position | Earlier benchmark and removal-trained reference; scene-only session (Table 6, Fig. 8) |
| `C379_L22_merge__correct_L34_FQ` | Merge candidates to 379 positions at L22 | Scene and question | Earlier benchmark only (Tables S30–S32) |
| `F67_L34_merge__correct_L34_readout` | Merge scene to 67 positions at L34 | Answer-readout position | Earlier benchmark only (Tables S30–S32) |
| `F0_L30_prune__correct_L34_CQ` | Remove scene positions at L30 | Candidate images and question | Removal-trained co-primary policy (Table 6) |
| `F0_L30_prune__correct_L34_readout` | Remove scene positions at L30 | Answer-readout position | Removal-trained comparator (Fig. 8) |
| `Fhalf_Chalf_L34_prune__correct_L34_readout` | Keep half of eligible scene and candidate positions at L34 | Answer-readout position | Removal-trained co-primary policy (Table 6) |

Merging configurations and the scene-and-question target appear only in the
earlier benchmark session, not in the paper's main text. The session crosswalk
is in the [experiment guide](../docs/experiments.md#intervention-checks-e4-gemma-only).

Cuts are zero-based post-block indices. Preserve each configuration's ordering
of correction and token operations using the released execution code.

## Training provenance

The custom participant-based training partition contains 505 Ego4D-source frames
with EgoHOS annotations, 88 participant clusters, and 1,010 paired-order requests.
It is not the original provider's train split. The 447 THU-READ, escape-room, and
YouTube evaluation frames were not used to train these weights. Reuse of that
evaluation set across practical experiments remains documented in the
[experiment guide](../docs/experiments.md).

The two matrices are new parameters, trained with the base model frozen,
using the code in `research/lifecycle_memory/application_pilot_steering.py`
and `decoder_token_matrix_steering.py`. In code, `steering` names the learned
activation correction. The files contain only those matrices
and configuration metadata, not source media or explicit sample identifiers.

## Loading and verification

From the repository root, in an environment with PyTorch and safetensors:

```python
from safetensors.torch import load_file
from research.lifecycle_memory.application_pilot_steering import LowRankSteering

path = "adapters/weights/native__correct_L34_CQ.seed17.final.safetensors"
adapter = LowRankSteering(hidden_size=5376, rank=4)
adapter.load_state_dict(load_file(path, device="cpu"), strict=True)
adapter.eval()
```

This loads the correction module, not the base VLM. Keep adapter parameters in
FP32; `LowRankSteering.forward` includes the recorded normalization and bounded
residual update. A plain matrix product is not an equivalent correction. Apply
it only to the configured positions using the corresponding execution path in
the [source guide](../docs/source_layout.md). The `legacy_driver_target` metadata
field is a compatibility label; the `positions` and `target` fields specify the
actual target. Deployment still requires the pinned base model, processor,
prompt layout, position mapping, and compatible runtime.

Verify all released checkpoint bytes without importing a model framework:

```bash
python -B -m unittest discover -s tests -p test_adapter_inventory.py -v
```

## Reuse and licensing

The CoRT code and adapter weights are released under [Apache-2.0](../LICENSE).
The model card of the exact [Gemma base-model revision](https://huggingface.co/google/gemma-4-31B-it/blob/3548789868c5356dbf307c98e6f609007b82b3eb/README.md)
states an Apache-2.0 license and also links the [Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license).
The weights are trained for this checkpoint, not for arbitrary Gemma models.

Training uses [Ego4D](https://ego4d.dev/request/ego4d) imagery and
[EgoHOS](https://github.com/owenzlz/EgoHOS) annotations. The publicly linked
[Ego4D agreement](https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf) retains
licensee ownership of derived software and models; the agreement accepted by
each user governs their dataset access. EgoHOS's repository MIT license is
worded for software; we found no separate annotation-specific statement about
checkpoint redistribution. This release does not relicense either dataset or
provide access to its imagery or annotations. See [data access](../docs/data_access.md).

The offline inventory exporter verifies the original safetensors headers,
parameter finiteness and content hashes without importing a model framework:

```bash
python analysis/export_adapter_inventory.py \
  --upstream /path/to/original-research-repository --include-weights
```

This maintainer command requires the original artifacts and checks every entry
against the existing provenance inventory before copying. It is not a download
command or a prerequisite for result replay. Without `--include-weights` it
exports metadata only, and `--output` must then point to a directory that has
no `weights/` folder; with the default output it stops with "Existing weights
require --include-weights".
