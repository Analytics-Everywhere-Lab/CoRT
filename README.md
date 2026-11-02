# CoRT: Counterfactual Role Tracing

Code, numerical results, and trained adapters accompanying the paper:

**CoRT: Counterfactual Role Tracing of Scene, Object, and Event-History Evidence Across Decoder Depth in Egocentric Video-Language Models**

## Overview

CoRT investigates how different input roles control the answers of
video-language models across decoder depth. The method replaces selected
hidden states between executions with different visual evidence, separating
scene, candidate-object, question, and event-history positions. The study
combines single-frame and video-clip experiments with evaluations of token
removal and learned activation corrections.

The repository provides experiment implementations, analysis configurations,
released numerical records, and trained correction adapters. The statistical
analyses can be reproduced on a CPU without dataset media or base-model weights.

## Repository Contents

| Component | Documentation |
| --- | --- |
| Method and position-role definitions | [Method](docs/method.md) and [implementation example](examples/README.md) |
| Experimental designs and analysis plans | [Experiments](docs/experiments.md) |
| Numerical results linked to manuscript tables and figures | [Results](results/README.md) and [paper-to-results index](results/paper_results_map.csv) |
| Statistical analysis and figure reproduction | [Reproduction](docs/reproduction.md) |
| Trained activation-correction adapters | [Adapter weights and usage](adapters/README.md) |
| Experiment source code and execution requirements | [Source layout](docs/source_layout.md) and [environment](docs/environment.md) |
| Dataset access and input preparation | [Data access](docs/data_access.md) and [preparation](docs/preparation.md) |

## Installation

The numerical analysis requires Python 3.12 or newer and the dependencies in
`requirements.txt`. It has been tested with Python 3.12.15 and 3.13.2.

```bash
git clone https://github.com/emadbazmandegan/cort_code.git
cd cort_code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the Results

Run the complete numerical analysis and generate the data-only figures:

```bash
python analysis/reproduce_all.py --figures
```

This command recomputes the supported statistics, compares them with the
released reference results, and generates data-only versions of Figures 3-8.
Successful completion reports that all four numerical replay groups passed.

| Output | Location |
| --- | --- |
| Recomputed statistics and comparison reports | `results/rebuilt/` |
| Generated figures | `figures/rendered/` |
| Table and figure evidence index | `results/paper_results_map.csv` |

The [reproduction guide](docs/reproduction.md) specifies which estimates and
intervals are recomputed and which are available as recorded summaries.
Detailed instructions cover the [supplementary single-frame analyses](docs/static_supplement_replay.md),
[temporal analyses](docs/temporal_replay.md), and
[practical evaluations](docs/practical_replay.md).

File integrity and release tests can be checked separately:

```bash
python scripts/verify_release.py
python -B -m unittest discover -s tests -v
```

## Experiments

The study evaluates Gemma4-31B (`google/gemma-4-31B-it`) and Qwen3.5-27B
(`Qwen/Qwen3.5-27B`) on egocentric visual-evidence tasks; revisions are pinned
in [Environment](docs/environment.md). CoRT combines matched evidence
counterfactuals, role-resolved state replacement, donor-consistent recovery, a
state-summary test, a role interaction decomposition and intervention checks;
the [method guide](docs/method.md) maps each component to its code.

| Experiment | Question | Models | Status |
| --- | --- | --- | --- |
| E0 | Does the correct scene help? Real, blank and wrong-time scenes | Both | Eligibility rule |
| E1 | Four-cut depth survey and screen | Both | Exploratory |
| E2 | Cut-pair test of the scene-to-complement reversal | Both | Gemma confirmatory; Qwen planned descriptive |
| T1 | History effect, depth order and V-only patch in clips | Gemma | Exploratory |
| E3 | All 15 unions of C, Q, P, R in single frames | Both | Descriptive; Qwen contrasts planned follow-up |
| T2 | All 15 unions in 17-frame clips | Gemma | Exploratory |
| E4 | Do token removal and learned correction behave as the replacement map predicts? Accuracy, latency, memory | Gemma | Development data (reused) |

The [experiment guide](docs/experiments.md) documents the designs, analysis
plans and a glossary of internal names.

## Trained Adapters

The release includes **16 rank-4 activation-correction adapters**, with
43,008 parameters each, in safetensors format. These adapters modify hidden
states after a decoder block while leaving the base-model weights unchanged.
They were trained for Gemma4-31B-it at the pinned revision and are used only in
the E4 intervention checks.
Checkpoint descriptions, loading instructions, and reuse conditions are
provided in the [adapter documentation](adapters/README.md).

## Data and Model Requirements

The numerical reproduction commands operate on released records and do not
rerun model inference. New inference or training experiments require authorized
dataset access, prepared inputs, the pinned model and processor revisions,
and a compatible GPU runtime. The source provides reference implementations;
GPU execution requires the preparation described in the
[execution guide](docs/source_layout.md).

Dataset images and videos, base-model weights, per-request model outputs, clip
labels and participant identity mappings are excluded from this repository. Dataset access procedures are described in
[Data Access](docs/data_access.md); model revisions and runtime requirements
are recorded in [Environment](docs/environment.md).

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

The code and trained adapter weights are released under the
[Apache License 2.0](LICENSE). Dataset and base-model terms apply separately;
see [data access](docs/data_access.md) and
[adapter licensing](adapters/README.md#reuse-and-licensing).
