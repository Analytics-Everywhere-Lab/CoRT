# What the code measures

Section, figure and equation numbers refer to the paper. Every statement about
a role holds within the fixed prompt order (Sec. 3.2).

## Components and code

Modules are under `research/lifecycle_memory/` unless a path is given. The
[source guide](source_layout.md) lists the runnable entrypoints, and the
[experiment guide](experiments.md) lists the E4 session runners.

| Paper component | Fig. 1 | Experiments | Main code |
| --- | --- | --- | --- |
| Matched evidence counterfactuals (Sec. 3.1) | a | all | Scene conditions `blank_full` and `wrong_time`: `build_phase2_reanchored_egohos_battery`. Clip conditions (B0, B1, B2, B4, B8, S1, S2, RV): `temporal_forecasting/protocol.py`, `recognition_protocol.py`, `followup_protocol.py` |
| Role partition (Sec. 3.2) | b | all | `patchaudit_component_partition` (copy in `protocols/specifications/component_partition.py`); clips: `partition_positions` in `temporal_forecasting/runtime.py` |
| Role-resolved state replacement (Sec. 3.3) | c | E1–E3, T1–T2 | `patchaudit_transport_operators` (copy in `protocols/specifications/transport_operators.py`); clips: `temporal_forecasting/runtime.py` |
| Donor-consistent recovery (Sec. 3.4) | c | E1–E3, T1–T2 | `analyze_patchaudit_transport_decomposition`, `analyze_patchaudit_heldout_component_closure`, `qwen_role_factorial`, `temporal_forecasting/analysis.py`; public replay: `analysis/replay_cluster_sufficient_stats.py`, `analysis/replay_temporal.py` |
| Depth tracing | d | E1, E2 | `run_patchaudit_transport_decomposition` (E1), `run_patchaudit_heldout_component_closure` (E2) |
| State-summary test (Sec. 3.3) | e | T1 | V-only patch in `temporal_forecasting/runtime.py`; analysis in `temporal_forecasting/dense_analysis.py` and `recognition_analysis.py` |
| Role interaction decomposition (Sec. 3.5) | f | E3, T2 | `factorial_component_analysis` in `analyze_patchaudit_heldout_component_closure` (Gemma), `qwen_role_factorial` (Qwen), `temporal_forecasting/followup_analysis.py` (clips) |
| Intervention checks (Sec. 4.4) | g | E4 | `decoder_cached_validation`, `role_aware_policy`, `role_aware_transfer`; the correction module is `LowRankSteering` in `application_pilot_steering` |

## Role partition

A single-frame input contains an instruction, a scene, two candidate-object
crops, a question with its output instruction, and the fixed answer prefix
`{"answer":"`. The processor converts this input into a sequence of positions,
and the decoder keeps one hidden-state vector at each position.

| Set | Meaning |
| --- | --- |
| `F` | Scene (context) image positions |
| `C` | Candidate-object image positions |
| `Q` | Question and output-instruction positions |
| `P` | Answer-prefix positions; the last one is the answer readout |
| `R` | Remaining instruction and template positions |
| `K` | All active positions outside `F` |

Thus `K = C union Q union P union R`. It includes the candidate images; it is
not a text-only set. Membership comes from the serialized prompt and processor
outputs. The code checks the sets rather than assuming a fixed number of
tokens for arbitrary inputs.

In clips, `F` divides into earlier-frame positions `H` and final-frame
positions `V`, so `A = H union V union K` disjointly, where `A` is the complete
active sequence. The single-frame and clip definitions must not be
interchanged.

## Matched evidence counterfactuals

A donor and a recipient differ in one evidence source only. In single frames
the scene is real, blank (all black; `blank_full`) or a different frame from
the same video (`wrong_time`), with candidates and label fixed. In clips the
counterfactual repeats the final frame in every scene slot (B4), so image
slots, prompt positions and final pixels stay identical. Behavioral controls
add the final or first frame alone, and shuffled or reversed orders, which are
scored for agreement with the original answer.

## Role-resolved state replacement

At a chosen cut, the operator copies the donor's rows at the selected
positions into the recipient run (Eq. 2). All other rows keep their recipient
values, and the remaining decoder blocks compute the answer score. Copying all
active positions reproduces the donor and serves as a positive control.

Cut indices are zero-based block outputs: `L22` is after block 22, meaning
23 blocks have completed. Replacement operates on the raw block output,
before the decoder's final normalization.

## Donor-consistent recovery

The answer margin is the teacher-forced difference of log2 probabilities of the
correct and incorrect one-token answers after the prefix, in bits (Eq. 3).
For each participant cluster, the code averages the absolute patched-to-donor
and recipient-to-donor margin differences over its rows, then averages over
clusters (Eq. 4):

```text
recovery(U) = 1 - D_U / D_0
```

A recovery of one means the patched and donor margins match. Zero means the
average discrepancy is unchanged. Negative values are possible, and overshooting
the donor margin is penalized. Recovery is a ratio of cluster means, not an
average of per-row ratios, and it measures donor closeness, not accuracy. In
clips, orders are averaged within clip and clips within participant, and all
clip estimates pass a guard of `D_0 > 0.01` bits.

## State-summary test

The state-summary test copies only the final-frame positions `V` from the
chronological clip into the repeated-final-frame run, or the reverse. Final
pixels are identical in the two runs, so a `V` patch transfers only what those
states absorbed from earlier frames. The criterion is descriptive, with no
threshold or equivalence test.

## Role interaction decomposition

The decomposition patches all 15 nonempty unions of `C`, `Q`, `P` and `R` at a
late cut. Möbius coefficients on an empty background (Eq. 5) ask whether
classes recover more together than separately, and Shapley allocations over
all 24 orders (Eq. 6) divide the complete-`K` recovery among the four classes.
A joint `C union Q` recovery and the sum of the `C` and `Q` Shapley allocations
are different quantities.

## Intervention checks

E4 tests token removal and learned rank-4 activation corrections directly. A
weak replacement effect is not itself a license to remove tokens. Accuracy and
request time are measured for the actual modified computation, against the
unmodified (native) model timed in the same session.

These measurements establish control under intervention, not native mediation.
The experiment records keep the distinctions between confirmation, planned
descriptive checks, planned follow-up contrasts, descriptive and exploratory
analyses, and development evaluation.
