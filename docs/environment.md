# Environment and Frozen Runtime

## Public Checks and Replay

The release targets Python 3.12 or newer and is tested on 3.12.15 and 3.13.2.
Syntax, JSON, inventory, and privacy-pattern checks use the standard library. Numerical replay requires
NumPy; figure generation additionally requires Matplotlib and Pillow.
`requirements.txt` pins NumPy 2.4.6, Matplotlib 3.10.9, and Pillow 11.3.0.

With dependencies provisioned, checks/replay need no model weights, datasets,
or network access:

```bash
python -B -m unittest discover -s tests -v
python -B analysis/replay_cluster_sufficient_stats.py
```

Offline installation requires a separately obtained local wheelhouse:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
```

The wheelhouse is not shipped. These dependencies do not supply the complete
historical model-run environment. Public checks do not validate GPU behavior,
original timing, dataset permissions, or checkpoint compatibility.

## Model and Processor Pins

| Model | Model repository and revision | Processor repository and revision |
| --- | --- | --- |
| Gemma4-31B (instruction-tuned) | `google/gemma-4-31B-it` at `3548789868c5356dbf307c98e6f609007b82b3eb` | Same repository, `3548789868c5356dbf307c98e6f609007b82b3eb` |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` at `fc05daec18b0a78c049392ed2e771dde82bdf654` | Same repository, `fc05daec18b0a78c049392ed2e771dde82bdf654` |

These are exact snapshots, not floating aliases. Model plans are in
`configs/static_cort/model_plans/`; the Qwen factorial config explicitly
pins both model and processor. `configs/temporal/gemma_model_inventory.json`
records individual snapshot-file SHA-256 values, including processor, tokenizer,
and native chat-template files. It contains metadata, not weights.
Model/processor terms remain independent of Apache-2.0 CoRT code.

## Prompt Serialization

`configs/static_cort/prompt_serialization.json` preserves exact source strings.
One user message contains leading text, scene, candidate A, candidate B, then
question. Despite its historical `SYSTEM_PROMPT` variable name, the leading
text is not a system-role message:

```text
Answer the first-versus-second-order target question using only the provided RGB inputs. Return exactly one JSON object with keys: answer, abstain, abstention_reason, confidence. The answer value must be A or B.
```

The source question is exactly:

```text
Input 1 is a full frame. Inputs 2 and 3 are unmarked target-only reference crops A and B. Which referenced target is the first-order interacting object, meaning the object directly manipulated by a visible hand rather than a second-order object interacting through it? Return exactly one JSON object with answer A or B plus the frozen abstention and confidence keys.
```

The literal uses `frozen`, not a paraphrase such as `fixed`. The native
chat template uses `add_generation_prompt=True`; the single-frame runner then
appends exactly `{"answer":"`, without added special tokens. A/B must be
distinct one-token continuations. Both scores come from the final prefix
position before either answer is appended. JSON completion is not the primary
outcome. A fully decoded native template stream was not separately saved;
snapshot, literals, assembly, and append operation specify serialization.

Clip serialization differs: one user message contains ordered scenes,
candidate A, candidate B, and the exact task prompt in the JSON protocol.
It uses the native generation prefix with `enable_thinking=False`, **without
the single-frame JSON prefix**. In clips, `P` is the native assistant-prefix
span that the chat template adds with `add_generation_prompt=True`; it ends at
the readout, and `Q` is the exact task prompt (`partition_positions` and the
template-boundary check in `temporal_forecasting/runtime.py`). Open/closed
native-template boundaries and contextual single-token A/B continuations are
checked. T2 keeps the original 17-frame task prompts for shuffles and reversal.

## Geometry and Metadata

Single-frame observed A/F/K counts are 921/266/655 for Gemma, 926/588/338 for Qwen.
C/Q/P/R counts are 512/75/3/65 and 200/73/3/62. Readout indices 920/925 are in
P and K, not Q. Cuts are zero-based decoder-block outputs: L22 follows 23 blocks.
Gemma survey cuts are 22/26/30/34, Qwen 24/28/32/36. Confirmation uses endpoints;
factorials use Gemma L34 and Qwen L36.

In clips, H is all earlier scene spans, V the final span, K every non-scene
position; H/V/K partition A. In the 17-frame clips H/V counts are 4,256/266; K
is 643 for new interaction, 623 for interaction order (A = 5,165/5,145). In the
9-frame current object clips H has 2,128 positions. Observed counts are audit targets, not replacements for deriving
membership from actual processor output.

Metadata records active masks; input/position IDs where present; image-slot
spans; exact question/prefix boundaries; readout; A/B token IDs; image/prompt/
model/processor hashes; donor/recipient alignment. F/C/Q/P/R are disjoint and
exhaustive; K = C union Q union P union R; CQR excludes readout. Candidate
pixels, order-specific identities, sequence geometry, and unselected tensor
rows remain fixed across matched interventions. Raw instance maps and evaluator
identities are not exported; see [Data Access](data_access.md).

## Historical Model Execution

The recorded stack used PyTorch `2.12.1+cu126`, Transformers `5.13.0`, BF16,
SDPA, offline checkpoints, and H100 workers at inference batch size one.
Clip runtime additionally imports Accelerate and Pillow plus Transformers
dependencies; preparation also needs image/decoding and, for pool allocation,
optimization dependencies. Versions of these additional packages were not recorded.
`requirements.txt` is not a lockfile for the historical container; exact
runtime/container identities remain bound in original receipts.

Gates cover model/processor identity, complete inventories, token/shape
alignment, finite scores, one-token scoring equivalence, self-patch invariance,
full-active closure, one firing per hook, unchanged unselected rows, and hook
removal. Single-frame self/full/token-equivalence tolerances are 0.001 bits;
clip controls also use 0.001 bits. T2 requires a new short technical check run
(`canary`), comparison with the T1 reference cells at tolerance 0.001,
unchanged dataset identity, and its 78-cell-per-clip inventory. Full gates/resource limits remain in
`protocols/contracts/` and source specifications.

## Export Identity

`manifests/source_to_export.json` records original-source and exported-byte
hashes separately. Only private account/container locations are replaced in
complete sanitized plans. Resources and technical gates stay unchanged.
Source comment and docstring edits are recorded separately in
`runtime/source_manifest.json`; they do not modify executable scientific statements.
Executed practical projections identify omitted per-request inventories and
nested records and are not complete launch configs. Embedded original hashes
still identify original artifacts, not sanitized bytes; export does not bypass
historical identity checks or authorize execution with missing private inputs.

`manifests/verification_report.json` is the historical record of the 1.0.0
release checks; its file hashes describe that release. The current checks are
`RELEASE_MANIFEST.json` and `manifests/replay_verification.json`. None of these
checks reruns GPU experiments.
