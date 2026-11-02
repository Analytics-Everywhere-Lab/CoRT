"""Prepare small immutable follow-up releases using already staged source assets."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .common import atomic_json, digest, read_json, read_jsonl, sha256
from .followup_protocol import CONDITIONS, followup_protocol, scene_permutation
from .protocol import inventory, protocol_contract, validate_dataset_protocol
from .release import prepare_release, verify_release


def prepare(repo, source_deploy, source_results, source_verification, data_root,
            design, human_validation, output, remote_root):
    """No SSH, GPU imports, label reads, source edits, or model copies."""
    repo, output = Path(repo).resolve(), Path(output)
    if output.exists():
        raise ValueError("choose a fresh follow-up deployment directory")
    remote_root = PurePosixPath(remote_root)
    if not remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("remote root must be an absolute canonical HPC path")
    prepared = {}
    for task in ("new_interaction", "interaction_order"):
        source = Path(source_deploy) / task
        sealed = verify_release(source / "release")
        base = read_json(source / "release" / "protocol.json")
        if base.get("task") != task + "_recognition":
            raise ValueError("source folder/task mismatch")
        receipt_path = Path(source_results) / task / "run-v2" / "full" / "receipt.json"
        receipt = read_json(receipt_path)
        verification = read_json(Path(source_verification) / f"{task}_full.json")
        local_data = Path(data_root) / f"{task}_v1_dataset"
        manifest = read_json(local_data / "manifest.json")
        if (verification.get("status") != "pass" or receipt.get("status") != "pass"
                or receipt.get("stage") != "recognition" or verification.get("stage") != "recognition"
                or verification.get("analysis", {}).get("recomputed") is not True
                or verification.get("receipt_sha256") != sha256(receipt_path)
                or verification.get("identity") != receipt.get("identity")
                or verification.get("result_set_sha256") != receipt.get("result_set_sha256")
                or receipt["identity"].get("release_sha256") != sealed["release_sha256"]
                or receipt["identity"].get("protocol_sha256") != digest(base)
                or receipt["identity"].get("dataset_manifest_sha256") != sha256(local_data / "manifest.json")):
            raise ValueError(f"unverified or mismatched source study: {task}")
        protocol = followup_protocol(base,
            source_dataset_manifest_sha256=sha256(local_data / "manifest.json"),
            source_result_set_sha256=receipt["result_set_sha256"],
            design_sha256=sha256(design), human_validation_sha256=sha256(human_validation))
        validate_dataset_protocol(manifest, protocol)
        public = read_jsonl(local_data / "public.jsonl")
        if sha256(local_data / "public.jsonl") != manifest["public_sha256"]:
            raise ValueError("source public file changed")
        tasks = inventory(public, protocol, "recognition")
        canary = inventory(public, protocol, "recognition-canary")
        config = read_json(source / "launch.json")
        if config["deployment"]["runtime"].get("binds"):
            raise ValueError("source launch has extra binds; review instead of carrying them forward")
        config = deepcopy(config)
        config.update(release_root=str(remote_root / task / "release"),
                      output_root=str(remote_root / task / "run-v1"))
        target = output / task
        target.mkdir(parents=True)
        atomic_json(target / "protocol.json", protocol)
        release = prepare_release(repo, target / "protocol.json", target / "release")
        permutations = [{"episode_id": row["episode_id"], "task": row["task"],
                         "permutations": {c: scene_permutation(row["task"], row["episode_id"], c)
                                          for c in CONDITIONS}} for row in public]
        atomic_json(target / "permutations.json", permutations)
        atomic_json(target / "launch.json", config)
        prepared[task] = {"release_sha256": release["release_sha256"],
            "protocol_sha256": digest(protocol), "source_identity": receipt["identity"],
            "source_result_set_sha256": receipt["result_set_sha256"],
            "permutations_sha256": sha256(target / "permutations.json"),
            "canary_tasks": len(canary), "full_tasks": len(tasks),
            "input_contract_unchanged": protocol_contract(base) == protocol_contract(protocol),
            "remote_release": config["release_root"], "remote_output": config["output_root"]}
    result = {"status": "prepared_not_submitted", "created_at": datetime.now(timezone.utc).isoformat(),
              "design_sha256": sha256(design), "human_validation_sha256": sha256(human_validation),
              "tasks": prepared, "model_execution": False, "evaluator_opened": False}
    atomic_json(output / "preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "source-deploy", "source-results", "source-verification", "data-root",
                "design", "human-validation", "output", "remote-root"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print({"status": result["status"], "tasks": {key: value["full_tasks"] for key, value in result["tasks"].items()}})


if __name__ == "__main__":
    main()
