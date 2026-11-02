"""Small synthetic example of row replacement; no model or dataset is loaded."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.lifecycle_memory.patchaudit_component_partition import build_component_partition


def replace_rows(recipient, donor, positions):
    if recipient.shape != donor.shape or recipient.ndim != 2:
        raise ValueError("Donor and recipient must have the same sequence-by-feature shape")
    indices = np.asarray(positions)
    if indices.size and not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Positions must be integer indices")
    indices = indices.astype(int)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("Positions must be a unique one-dimensional index list")
    if np.any(indices < 0) or np.any(indices >= recipient.shape[0]):
        raise ValueError("Position is outside the sequence")
    patched = recipient.copy()
    patched[indices] = donor[indices]
    return patched


def main():
    # Positions and states below are synthetic, not tokens or results from the paper.
    partition = build_component_partition(
        sequence_length=18, active_positions=tuple(range(18)),
        image_slot_positions=((2, 3, 4, 5), (8, 9), (12, 13)),
        literal_user_prompt_start=14, literal_user_prompt_count=2,
        answer_prefix_token_count=2,
    )
    full_scene = np.array([2, 3, 4, 5])
    complement = np.setdiff1d(np.arange(18), full_scene)
    recipient = np.zeros((18, 4), dtype=np.float64)
    donor = np.arange(72, dtype=np.float64).reshape(18, 4)
    patched = replace_rows(recipient, donor, full_scene)
    assert np.array_equal(patched[full_scene], donor[full_scene])
    assert np.array_equal(patched[complement], recipient[complement])
    assert partition.full_frame.count == len(full_scene)
    assert partition.complement.count == len(complement)
    print(json.dumps({"example": "synthetic_row_replacement", "state_shape": [18, 4],
                      "scene_rows": full_scene.tolist(), "complement_rows": complement.tolist(),
                      "donor_rows_copied": len(full_scene),
                      "recipient_rows_preserved": len(complement)}, indent=2))


if __name__ == "__main__":
    main()
