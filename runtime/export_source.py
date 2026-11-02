"""Offline, bounded source-reference export; never prepares an experiment release.

Uses the repository's release selectors as provenance, then closes the explicit
scientific roots under Python imports. Only four legacy helpers are symbol-sliced.
Their selected definitions are copied from source, not rewritten or reimplemented.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import sys
import tokenize
import zipfile

PACKAGE = "research/lifecycle_memory/"
JOBS = "docs/research/lifecycle_memory/phase 2 research/jobs/"
JOB0 = JOBS + "job0/run_phase1_qwen27b_hf.py"
PRIVATE_SELECTION = JOBS + "patchaudit_jobE_heldout_component_closure/heldout_selection.json"
CONSTRUCTION_ARCHIVE = (
    "papers/patchaudit_final_revision_2026_09_08/packaging/"
    "patchaudit_anonymous_scientific_supplement_2026_09_08_v2.zip")
CONSTRUCTION_PREFIX = "protocols/construction/"
PREPARATION_MODULES = ("build_phase2_reanchored_egohos_battery", "temporal_dense_current_data")
ROOT_MODULES = """
run_phase2_reanchored_behavior analyze_phase2_reanchored_behavior
run_phase2_reanchored_layer_capture analyze_phase2_reanchored_layer_capture
run_phase2_reanchored_causal_patch analyze_phase2_reanchored_causal_patch
run_phase2_reanchored_answer_token_causal_patch
analyze_phase2_reanchored_answer_token_causal_patch
analyze_phase2_reanchored_answer_token_confirmation
run_patchaudit_transport_decomposition analyze_patchaudit_transport_decomposition
verify_patchaudit_transport_metadata run_patchaudit_heldout_component_closure
analyze_patchaudit_heldout_component_closure verify_patchaudit_heldout_component_metadata
build_patchaudit_jobE_heldout_selection patchaudit_component_partition
patchaudit_transport_operators qwen_role_factorial qwen_role_factorial_release
role_aware_policy run_role_aware_policy role_aware_policy_release
role_aware_killarney_runtime role_aware_killarney_release
role_aware_transfer run_role_aware_transfer role_aware_transfer_release
""".split() + list(PREPARATION_MODULES)
TEST_MODULES = """
test_run_phase2_reanchored_behavior test_phase2_reanchored_causal_patch
test_phase2_reanchored_answer_token_causal_patch test_reanchored_answer_token_confirmation
test_patchaudit_transport_operators test_patchaudit_transport_decomposition
test_patchaudit_component_partition test_patchaudit_heldout_component_closure
test_qwen_role_factorial test_role_aware_policy test_role_aware_killarney_runtime
test_role_aware_transfer test_temporal_forecasting_runtime
test_temporal_recognition_runtime test_temporal_recognition_analysis
test_temporal_forecasting_protocol test_temporal_forecasting_data
test_temporal_followup_analysis test_temporal_followup_protocol
test_temporal_recognition_launch test_temporal_recognition_protocol
""".split()
REDUCED = {PACKAGE + name + ".py" for name in (
    "protocol_v3_3rscan_visual_vlm", "build_epic_visor_dynamic_smoke",
    "run_epic_visor_dynamic_vlm_smoke", "run_epic_visor_dynamic_llamacpp_metal")}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module_name(name):
    return name.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def import_records(name, tree):
    """Yield base module and requested names, resolving relative package imports."""
    package = module_name(name).split(".")
    if not name.endswith("/__init__.py"):
        package.pop()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package[:len(package) - node.level + 1]
                                + ([base] if base else []))
            yield base, [alias.name for alias in node.names]


def local_paths(root, module):
    path = module.replace(".", "/")
    return [name for name in (path + ".py", path + "/__init__.py")
            if (root / name).is_file()]


def bindings(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import)
                            else a.name) for a in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for target in targets for n in ast.walk(target)
                if isinstance(n, ast.Name)}
    return set()


def slice_helper(source, wanted):
    tree = ast.parse(source)
    nodes = tree.body
    owners = {key: i for i, node in enumerate(nodes) for key in bindings(node)}
    assert wanted <= owners.keys(), f"unknown helper symbols: {wanted - owners.keys()}"
    kept, pending = set(), set(wanted)
    while pending:
        key = pending.pop()
        index = owners.get(key)
        if index is None or index in kept:
            continue
        kept.add(index)
        pending.update(n.id for n in ast.walk(nodes[index])
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    lines = source.splitlines(keepends=True)
    output = ['"""Compatibility helpers retained verbatim for the public source closure.\n',
              'Unrelated legacy commands and API backends are not part of this export.\n"""\n']
    retained = set()
    for i, node in enumerate(nodes):
        future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if i not in kept and not future:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        output.append("".join(lines[start - 1:node.end_lineno]) + "\n\n")
        retained.update(bindings(node))
    result = "".join(output)
    # Compare each retained top-level statement, including imports and constants.
    exported = ast.parse(result).body[1:]
    expected = [n for i, n in enumerate(nodes) if i in kept
                or isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert [ast.dump(n) for n in exported] == [ast.dump(n) for n in expected]
    return result, sorted(retained)


def closure(root, roots):
    selected = set(roots)
    demand = {name: set() for name in REDUCED}
    rendered, slices = {}, {}
    while True:
        before = (set(selected), {k: set(v) for k, v in demand.items()})
        for name in sorted(selected):
            source = (root / name).read_text()
            if name in REDUCED:
                source, slices[name] = slice_helper(source, demand[name])
            rendered[name] = source
            for base, symbols in import_records(name, ast.parse(source)):
                for target in local_paths(root, base):
                    selected.add(target)
                    if target in REDUCED:
                        assert symbols is not None and "*" not in symbols, (
                            f"cannot safely reduce module/star import: {name}: {base}")
                        demand[target].update(symbols)
                for symbol in symbols or []:
                    selected.update(local_paths(root, base + "." + symbol))
                parts = base.split(".")
                for i in range(1, len(parts) + 1):
                    init = "/".join(parts[:i]) + "/__init__.py"
                    if (root / init).is_file():
                        selected.add(init)
        if before == (selected, demand):
            return rendered, slices, demand


def sanitize(text):
    """Redact machine-specific prefixes, not scientific constants or hash pins."""
    changes = []
    rules = (
        (r"/(?:Users|home)/[^/\s\"'<>]+(?:/codex_home/mani-research)?", "/path/to/cort"),
        (r"/scratch/[a-z]/[^/\s\"'<>]+", "/path/to/scratch"),
        (r"/(?:project|projects)/[0-9]+/[^/\s\"'<>]+", "/path/to/project"),
        (r"(?<![\w-])(?:aip|rrg|def)-[a-zA-Z][a-zA-Z0-9_-]*", "REPLACE_WITH_ACCOUNT"),
    )
    for pattern, replacement in rules:
        text, count = re.subn(pattern, replacement, text)
        if count:
            changes.append({"kind": "public_path_or_account", "replacement": replacement,
                            "occurrences": count})
    return text, changes


def comment_spans(source):
    """Return complete comment and docstring spans in source-character offsets."""
    lines = io.StringIO(source).readlines()
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append((offsets[token.start[0] - 1] + token.start[1],
                          offsets[token.end[0] - 1] + token.end[1]))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                value = node.body[0]
                # AST columns are UTF-8 byte offsets; tokenize columns are characters.
                start = len(lines[value.lineno - 1].encode()[:value.col_offset].decode())
                end = len(lines[value.end_lineno - 1].encode()[:value.end_col_offset].decode())
                spans.append((offsets[value.lineno - 1] + start,
                              offsets[value.end_lineno - 1] + end))
    return sorted(spans)


def executable_ast(source):
    """Normalize Python syntax while excluding documentation strings."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def apply_comment_edits(source, changes):
    """Replay recorded editorial edits without modifying executable statements."""
    original = executable_ast(source)
    for change in changes:
        if change.get("kind") != "public_comment_edit":
            continue
        before, after = change["before"], change["after"]
        if sanitize(after)[0] != after:
            raise ValueError("Editorial text contains a private machine path or account")
        matches = [(start, end) for start, end in comment_spans(source)
                   if source[start:end] == before]
        if len(matches) != 1:
            raise ValueError("Editorial edit must match one complete comment or docstring")
        start, end = matches[0]
        source = source[:start] + after + source[end:]
        if executable_ast(source) != original:
            raise ValueError("Editorial edit changed executable Python syntax")
    return source


def builder_candidates(root):
    """Reuse upstream selectors without copying their bulk inputs or whole repo."""
    sys.path.insert(0, str(root))
    inventories = {}
    for name in ("qwen_role_factorial_release", "role_aware_transfer_release"):
        module = importlib.import_module("research.lifecycle_memory." + name)
        inventories[name] = module.source_files(root)
    temporal = importlib.import_module("research.lifecycle_memory.temporal_forecasting.release")
    inventories["temporal_forecasting.release"] = temporal._source_paths(root)
    return inventories


def construction_documents(root):
    """Project four archived records; never publish the hash-bound runtime pack."""
    archive = root / CONSTRUCTION_ARCHIVE
    archive_sha256 = sha(archive.read_bytes())
    prefix = "provenance/construction_chronology/"
    names = ("selection_summary.json", "HASH_CROSSWALK.json", "contract_extract.json",
             "execution_binding_extract.json")
    with zipfile.ZipFile(archive) as bundle:
        raw = {name: bundle.read(prefix + name) for name in names}
    values = {name: json.loads(value) for name, value in raw.items()}
    selection = values["selection_summary.json"]
    original = (root / PRIVATE_SELECTION).read_bytes()
    record = json.loads(original)
    for key, count in (("frame_unit_ids", "frame_count"), ("request_ids", "request_count")):
        assert len(record["selected"].pop(key)) == record["selected"][count]
    assert sha(original) == selection["original_sha256"], "archived selection hash mismatch"
    assert record == selection["original_record"], "archived selection summary drift"

    crosswalk = values["HASH_CROSSWALK.json"]
    for item in crosswalk["construction_source"]:
        # Module names describe code, not source-media locators or private paths.
        item["module"] = Path(item.pop("path")).stem
    crosswalk["retained_anonymized_specification"].pop("path")
    crosswalk["scope"] = (
        "Original hashes identify original records, not these documentary extracts. "
        "Archived construction source hashes are not hashes of the current source export.")
    contract = values["contract_extract.json"]
    contract_record = contract["original_record_with_redactions"]
    contract_record.pop("outputs")
    contract_record["prespec"].pop("path")
    contract.pop("path_mapping")
    contract["omitted_fields"].extend(("outputs", "prespec.path"))
    bindings = values["execution_binding_extract.json"]
    artifact_roles = {
        "patchaudit_heldout_component_closure_v1_2026_07_26.json": "prespecification",
        "heldout_selection.json": "private_execution_selection",
        "heldout_public_requests.jsonl": "private_execution_requests",
    }
    for release in bindings["releases"]:
        for item in release["selected_entries"]:
            item["artifact_role"] = artifact_roles[item.pop("artifact")]

    documents = {}
    for name, value in values.items():
        value.update(
            kind="documentary_construction_extract", not_an_execution_manifest=True,
            provenance={"archive_label": "scientific_supp_v2",
                        "archive_sha256": archive_sha256,
                        "archive_member_sha256": sha(raw[name])},
            limitations=("Recorded local chronology and content consistency only; not proof of "
                         "independent blinding or externally registered freeze timing. "
                         "Execution requires an independently obtained private execution manifest "
                         "and its real hash-bound inputs; this extract cannot replace them."))
        output = "heldout_selection_documentary_summary.json" if name == "selection_summary.json" else name.lower()
        documents[CONSTRUCTION_PREFIX + output] = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return documents


def export(root, destination):
    assert root != destination and root not in destination.parents
    prior_manifest = Path(__file__).with_name("source_manifest.json")
    editorial = {}
    if prior_manifest.exists():
        for entry in json.loads(prior_manifest.read_text())["files"]:
            editorial[entry["path"]] = [c for c in entry["changes"]
                                       if c["kind"] == "public_comment_edit"]
    inventories = builder_candidates(root)
    roots = {PACKAGE + n + ".py" for n in ROOT_MODULES}
    roots.update(PACKAGE + "tests/" + n + ".py" for n in TEST_MODULES)
    roots.update(p.relative_to(root).as_posix()
                 for p in (root / PACKAGE / "temporal_forecasting").glob("*.py"))
    roots.add(JOB0)
    sources, slices, demand = closure(root, roots)
    assert all(name.startswith((PACKAGE, JOBS)) for name in sources)
    # Compatibility configs remain at the original paths used by runtime code.
    configs = {name for names in inventories.values() for name in names
               if name.startswith(JOBS) and name.endswith(".json")}
    # Do not include data/labels/manifests from old prepared experiment folders.
    configs = {name for name in configs if "/artifacts/" not in name and name != PRIVATE_SELECTION}
    assert not (destination / PRIVATE_SELECTION).exists(), (
        "Remove the previously published private heldout_selection.json before exporting; "
        "the documentary summary is not its runtime replacement.")
    for name in sorted(configs):
        sources[name] = (root / name).read_text()
    entries = []
    for name, source in sorted(sources.items()):
        original = (root / name).read_bytes()
        source, changes = sanitize(source)
        if editorial.get(name):
            source = apply_comment_edits(source, editorial[name])
            changes.extend(editorial[name])
        if name in slices:
            changes.insert(0, {"kind": "ast_helper_subset", "requested_symbols": sorted(demand[name]),
                               "retained_symbols": slices[name], "retained_ast_equal": True})
        if name.endswith(".py"):
            ast.parse(source, filename=name)
            compile(source, name, "exec")
        elif name.endswith(".json"):
            json.loads(source)
        target = destination / name
        assert not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.encode()
        target.write_bytes(raw)
        entries.append({"path": name, "original_path": name, "original_sha256": sha(original),
                        "export_sha256": sha(raw), "bytes": len(raw), "changes": changes,
                        "selection": "root" if name in roots else "compatibility_config" if name in configs
                        else "transitive_import"})
    # Remove stale Python copies only within the managed research source tree.
    stale = [p for p in (destination / "research").rglob("*.py")
             if p.relative_to(destination).as_posix() not in sources]
    for path in stale:
        path.unlink()
    manifest = {"schema_version": 1, "kind": "public_source_reference",
        "not_a_prepared_release": True, "network_used": False,
        "algorithm_policy": "Executable scientific statements preserved; public paths/accounts and comment/docstring edits are recorded separately; legacy helper statements AST-equal before editorial edits.",
        "builder_selection": {name: {"candidate_count": len(names),
            "selected_count": len(set(names) & sources.keys()),
            "excluded_paths": sorted(set(names) - sources.keys())}
            for name, names in inventories.items()},
        "roots": sorted(roots), "files": entries,
        "generated_files": [],
        "python_files": sum(e["path"].endswith(".py") for e in entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "removed_stale_source_paths": sorted(p.relative_to(destination).as_posix() for p in stale),
        "validation_scope": "CPU source/import/synthetic checks only; no dataset, model, GPU, scheduler or new experiment execution."}
    for name, source in construction_documents(root).items():
        path = destination / name
        assert not path.is_symlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = source.encode()
        path.write_bytes(raw)
        manifest["generated_files"].append({"path": name, "original_path": None,
            "original_sha256": None, "export_sha256": sha(raw), "bytes": len(raw),
            "changes": [{"kind": "sanitized_archived_construction_documentary_projection"}]})
    for name in ("runtime/export_source.py", "runtime/verify_source.py", "runtime/cpu_checks.py",
                 "runtime/requirements-cpu.txt", "runtime/requirements-gpu-reference.txt",
                 "tests/test_source.py", "docs/source_layout.md", "docs/preparation.md",
                 "protocols/construction/README.md"):
        path = destination / name
        raw = path.read_bytes()
        manifest["generated_files"].append({"path": name, "original_path": None,
            "original_sha256": None, "export_sha256": sha(raw), "bytes": len(raw),
            "changes": [{"kind": "new_public_validation_or_documentation"}]})
    target = destination / "runtime/source_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("python_files", "total_bytes")}, sort_keys=True))
    print(f"Exported {len(entries)} files; {len(slices)} reduced legacy helpers; {len(stale)} stale copies removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", "--source", dest="upstream", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    export(args.upstream.resolve(strict=True), args.destination.resolve(strict=True))
