"""Compare fresh reference cells against a verified prior recognition run."""

import argparse
import math
from pathlib import Path

from .common import atomic_json, digest, read_json, sha256
from .followup_protocol import validate_followup_protocol


def _verified(run, report):
    receipt_path = Path(run) / "receipt.json"
    receipt, checked = read_json(receipt_path), read_json(report)
    if (checked.get("status") != "pass" or receipt.get("status") != "pass"
            or checked.get("identity") != receipt.get("identity")
            or checked.get("receipt_sha256") != sha256(receipt_path)
            or checked.get("result_set_sha256") != receipt.get("result_set_sha256")):
        raise ValueError("missing or stale independent verification")
    rows = [read_json(path) for path in (Path(run) / "results").glob("*/*.json")]
    for row in rows:
        if (row.get("identity") != receipt["identity"]
                or row.get("result_sha256") != digest({k: v for k, v in row.items() if k != "result_sha256"})):
            raise ValueError("reference row identity/hash mismatch")
    if len(rows) != receipt["tasks"] or digest(sorted(r["result_sha256"] for r in rows)) != receipt["result_set_sha256"]:
        raise ValueError("reference result inventory mismatch")
    return receipt, rows


def _key(row):
    if row["kind"] == "behavior" and row["condition"] in ("B0", "B4"):
        return (row["episode_id"], row["order"], "behavior", row["condition"])
    if (row["kind"] == "patch" and row["cut"] == 34
            and row["position_set"] in ("K", "C+Q+P+R", "A")):
        return (row["episode_id"], row["order"], "patch", row["donor_condition"],
                row["recipient_condition"], "K" if row["position_set"] != "A" else "A")
    return None


def compare_references(source_run, source_report, followup_run, followup_report, protocol):
    protocol = validate_followup_protocol(read_json(protocol))
    old, old_rows = _verified(source_run, source_report)
    new, new_rows = _verified(followup_run, followup_report)
    if old["stage"] != "recognition" or old["result_set_sha256"] != protocol["source_result_set_sha256"]:
        raise ValueError("wrong original study")
    if new["identity"]["protocol_sha256"] != digest(protocol):
        raise ValueError("wrong follow-up protocol")
    for key in ("public_sha256", "evaluator_sha256", "dataset_manifest_sha256", "model_inventories",
                "runtime_spec_sha256", "runtime_identity"):
        if old["identity"][key] != new["identity"][key]:
            raise ValueError(f"historical reference identity differs: {key}")
    indexed = {_key(row): row for row in old_rows if _key(row) is not None}
    selected = [row for row in new_rows if _key(row) is not None]
    if len(selected) != 12 * len({row["episode_id"] for row in new_rows}):
        raise ValueError("incomplete fresh B0/B4/K/A comparison inventory")
    maximum, failures = 0., []
    for row in selected:
        old_row = indexed.get(_key(row))
        if old_row is None:
            raise ValueError("missing historical reference cell")
        for field in ("a_logprob", "b_logprob", "margin_ab_bits"):
            values = row[field], old_row[field]
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError("nonfinite reference score")
            delta = abs(values[0] - values[1])
            maximum = max(maximum, delta)
            if delta > .001:
                failures.append({"task_id": row["task_id"], "field": field, "absolute_difference": delta})
    return {"status": "fail" if failures else "pass", "compared_cells": len(selected),
            "tolerance": .001, "max_absolute_difference": maximum, "failures": failures,
            "protocol_sha256": digest(protocol), "stage": new["stage"],
            "followup_receipt_sha256": sha256(Path(followup_run) / "receipt.json"),
            "source_result_set_sha256": old["result_set_sha256"],
            "followup_result_set_sha256": new["result_set_sha256"], "labels_opened": False}


def require_canary_reference(protocol, deployment, canary):
    """A changed, missing, or failed historical check cannot authorize full work."""
    path = deployment.get("followup_reference_check")
    if not path or not Path(path).is_file():
        raise ValueError("follow-up needs the passing historical canary comparison")
    checked = read_json(path)
    if (sha256(path) != deployment.get("followup_reference_check_sha256")
            or checked.get("status") != "pass" or checked.get("failures") != []
            or checked.get("stage") != "recognition-canary" or checked.get("compared_cells") != 48
            or checked.get("tolerance") != .001 or checked.get("labels_opened") is not False
            or type(checked.get("max_absolute_difference")) not in (int, float)
            or not 0 <= checked["max_absolute_difference"] <= .001
            or checked.get("source_result_set_sha256") != protocol["source_result_set_sha256"]
            or checked.get("protocol_sha256") != digest(protocol)
            or checked.get("followup_result_set_sha256") != canary.get("result_set_sha256")
            or checked.get("followup_receipt_sha256") != sha256(deployment["canary_receipt"])):
        raise ValueError("historical canary comparison is failed, stale, or not bound to this follow-up")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-run", "source-report", "followup-run", "followup-report", "protocol", "output"):
        parser.add_argument("--" + key, required=True)
    args = vars(parser.parse_args())
    output = Path(args.pop("output"))
    if output.exists():
        raise ValueError("comparison report already exists")
    result = compare_references(**args)
    atomic_json(output, result)
    print({k: result[k] for k in ("status", "compared_cells", "max_absolute_difference")})
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
