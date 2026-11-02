import math
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest

from research.lifecycle_memory.temporal_forecasting.common import atomic_json, digest
from research.lifecycle_memory.temporal_forecasting.execution import run_worker, collect, canary_admission
from research.lifecycle_memory.temporal_forecasting.protocol import (
    dense_current_protocol, protocol_contract, inventory, assign_tasks,
)
from research.lifecycle_memory.temporal_forecasting.analysis import analyze
from research.lifecycle_memory.temporal_forecasting.dense_analysis import validate_dense_technical


def score(margin):
    odds = margin * math.log(2)
    a = -math.log1p(math.exp(-odds))
    b = -math.log1p(math.exp(odds))
    return {"a_logprob": a, "b_logprob": b, "margin_ab_bits": margin}


class FakeDenseBackend:
    calls = 0

    def __init__(self, model, path, *, contract, **kwargs):
        self.binding = {"task": contract["task"], "contract_sha256": digest(contract)}

    @staticmethod
    def native(order, condition):
        return score((1 if order == 0 else -1) * {"B0": 4., "B1": 2., "B4": 1., "B8": 3.}[condition])

    def score(self, episode, order, condition):
        type(self).calls += 1
        return {**self.native(order, condition), **self.binding, "elapsed_s": .01}

    def patch(self, episode, order, donor, recipient, cut, positions, self_patch=False):
        type(self).calls += 1
        d, r = self.native(order, donor), self.native(order, recipient)
        fraction = {"H": .8, "V": .2, "K": .5, "A": 1.}[positions]
        value = r["margin_ab_bits"] + fraction * (d["margin_ab_bits"] - r["margin_ab_bits"])
        audit = {"hook_calls": 1, "selected_rows_equal": True, "preserved_rows_equal": True}
        def certificate(native):
            return {"passed": True, "scores": native, "max_drift_bits": 0.,
                    "absolute_error_bits": dict.fromkeys(("a_logprob", "b_logprob", "margin_ab_bits"), 0.)}
        return {**score(value), **self.binding, "elapsed_s": .02,
                "donor_margin_ab_bits": d["margin_ab_bits"], "recipient_margin_ab_bits": r["margin_ab_bits"],
                "donor_score": d, "recipient_score": r,
                "technical": {"alignment_pass": True, "self_patch_pass": True, "full_A_pass": True,
                              "full_forward_reference": True, "suffix_replay": False,
                              "forward_count": 6, "self_max_drift_bits": 0.,
                              "full_closure_max_drift_bits": 0., "tolerance_bits": .001,
                              "controls_reused": dict.fromkeys(("recipient_self", "donor_self", "full_A"), False),
                              **audit},
                "controls": {"recipient_self": certificate(r), "donor_self": certificate(d),
                             "full_A": {**certificate(d), **audit},
                             "selected_self": certificate(r) if self_patch else None}}


class DenseExecutionTests(unittest.TestCase):
    def test_canary_inventory_worker_resume_and_descriptive_analysis(self):
        policy = {"version": 1, "kind": "matched_static_reuse",
                  "historical_static_sha256": "a" * 64, "temporal_exposure_sha256": "b" * 64}
        protocol = dense_current_protocol(policy)
        contract = protocol_contract(protocol)
        rows = [{"episode_id": f"e{i}", "cluster_id": f"c{i}", "split": "pilot",
                 "task": contract["task"], "contract_sha256": digest(contract)} for i in range(4)]
        labels = [{"episode_id": r["episode_id"], "correct_candidate": 0, "label_status": "audited",
                   "static_label_preserved": True, "static_source_sha256": "c" * 64} for r in rows]
        tasks = inventory(rows, protocol, "current-canary")
        self.assertEqual(len(tasks), 176)
        identity = {"simulation": True, "release_sha256": "fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {"run_dir": tmp, "identity": identity, "contract": contract,
                       "assignments": assign_tasks(tasks, 4), "episodes": rows,
                       "model_paths": {t["model_key"]: "/not-a-model" for t in tasks},
                       "attention_backend": "sdpa", "deadline_epoch": time.time() + 60}
            path = root / "worker_context.json"
            atomic_json(path, context)
            FakeDenseBackend.calls = 0
            for rank in range(4):
                self.assertEqual(run_worker(path, rank, FakeDenseBackend)["status"], "pass")
            self.assertEqual(FakeDenseBackend.calls, 176)
            for rank in range(4):
                run_worker(path, rank, FakeDenseBackend)
            self.assertEqual(FakeDenseBackend.calls, 176)
            results = collect(root, tasks, identity)
            self.assertEqual(validate_dense_technical(results)["status"], "pass")
            drifted = deepcopy(results)
            behavior = next(row for row in drifted if row["kind"] == "behavior" and row["condition"] == "B0")
            behavior["a_logprob"] -= .01
            behavior["b_logprob"] -= .01
            self.assertEqual(validate_dense_technical(drifted)["status"], "fail")
            result = analyze(results, labels, protocol, "current-canary", draws=99)
            self.assertTrue(result["analysis_complete"])
            self.assertEqual(result["family_size"], 0)
            self.assertAlmostEqual(result["patch_profiles"][0]["recovery"]["estimate"], .8)
            with self.assertRaisesRegex(ValueError, "simulation"):
                canary_admission({"status": "pass", "stage": "current-canary", "identity": identity},
                                 identity, tasks, 4, 3600)


if __name__ == "__main__":
    unittest.main()
