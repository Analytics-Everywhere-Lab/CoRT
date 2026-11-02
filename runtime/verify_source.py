"""Verify source hashes, local import closure and optional upstream provenance."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.export_source import (
    CONSTRUCTION_PREFIX, JOB0, PRIVATE_SELECTION, REDUCED, bindings,
    apply_comment_edits, construction_documents, import_records, local_paths, sanitize, slice_helper,
)

HEAVY = {"torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "mlx"}
SCIPY_MODULES = {
    "build_phase2_reanchored_egohos_battery", "phase2_job3_egohos_burned_items",
    "phase2_job3_egohos_census", "phase2_job3_marker_renderer",
}


def missing_optional_dependencies(module):
    if module.rsplit(".", 1)[-1] in SCIPY_MODULES and importlib.util.find_spec("scipy") is None:
        return ["scipy"]
    return []


class BlockModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in HEAVY:
            raise ImportError("Model backend imports forbidden in CPU source smoke: " + fullname)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(tree):
    # Includes imports/constants under module-level try/if, but no function locals.
    names = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        names.update(bindings(node))
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            pending.extend(getattr(node, "body", []))
            pending.extend(getattr(node, "orelse", []))
            pending.extend(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                pending.extend(handler.body)
    return names


def check_imports(root, files):
    trees = {name: ast.parse((root / name).read_text(), filename=name)
             for name in files if name.endswith(".py")}
    external, count = set(), 0
    for name, tree in trees.items():
        compile(tree, name, "exec")
        for base, symbols in import_records(name, tree):
            if not base.startswith(("research", "runtime")):
                external.add(base.split(".")[0])
                continue
            count += 1
            paths = local_paths(root, base)
            folder = root / base.replace(".", "/")
            require(paths or folder.is_dir(), f"unresolved local module: {name}: {base}")
            names = set().union(*(definitions(trees[p]) for p in paths if p in trees))
            for symbol in symbols or []:
                if symbol == "*":
                    require(paths and not REDUCED.intersection(paths),
                            f"star import must target a complete module: {name}: {base}")
                    continue
                require(symbol in names or local_paths(root, base + "." + symbol),
                        f"unresolved local symbol: {name}: {base}.{symbol}")
    return {"local_import_statements": count,
            "external_import_roots": sorted(external - sys.stdlib_module_names)}


def verify(root=ROOT, upstream=None):
    manifest = json.loads((root / "runtime/source_manifest.json").read_text())
    entries = manifest["files"] + manifest.get("generated_files", [])
    files = {e["path"] for e in entries}
    require(len(files) == len(entries), "duplicate manifest path")
    require(manifest["kind"] == "public_source_reference" and manifest["not_a_prepared_release"],
            "source-reference status missing")
    require(PRIVATE_SELECTION not in files and not (root / PRIVATE_SELECTION).exists(),
            "private held-out runtime selection must not be published")
    prefixes = ("research/", "docs/research/", CONSTRUCTION_PREFIX)
    physical = {p.relative_to(root).as_posix() for top in prefixes
                for p in (root / top).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts}
    require(physical == {p for p in files if p.startswith(prefixes)},
            f"unmanifested or missing source: {sorted(physical ^ files)}")
    for entry in entries:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe manifest path")
        path = root / relative
        require(not path.is_symlink() and root.resolve() in path.resolve().parents, "unsafe source path")
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["export_sha256"],
                f"export hash/size drift: {relative}")
        if upstream is not None and entry["original_path"] is not None:
            original = upstream / entry["original_path"]
            require(digest(original) == entry["original_sha256"], f"original hash drift: {relative}")
            source = original.read_text()
            if str(relative) in REDUCED:
                change = next(c for c in entry["changes"] if c["kind"] == "ast_helper_subset")
                source, retained = slice_helper(source, set(change["requested_symbols"]))
                require(retained == change["retained_symbols"], "helper symbol drift")
            expected, _ = sanitize(source)
            if relative.suffix == ".py":
                expected = apply_comment_edits(expected, entry["changes"])
            require(expected == path.read_text(), f"undocumented source edit: {relative}")
    if upstream is not None:
        for name, expected in construction_documents(upstream).items():
            require(name in files and (root / name).read_text() == expected,
                    f"documentary construction projection drift: {name}")
    require(JOB0 in files, "dynamic job0 model helper missing")
    report = check_imports(root, files)
    report.update(python_files=manifest["python_files"],
                  manifest_files=len(manifest["files"]),
                  generated_files=len(manifest.get("generated_files", [])),
                  upstream_verified=upstream is not None)
    return report


def smoke(root=ROOT):
    """Import CPU-safe modules; keep Torch-backed steering as syntax-only."""
    sys.meta_path.insert(0, BlockModelImports())
    manifest = json.loads((root / "runtime/source_manifest.json").read_text())
    imported, source_only, optional_skips = [], [], []
    for entry in manifest["files"]:
        name = entry["path"]
        if not name.startswith("research/") or not name.endswith(".py") or "/tests/" in name:
            continue
        module = name.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        if module.rsplit(".", 1)[-1] in {"application_pilot_steering", "decoder_token_matrix_steering"}:
            source_only.append(module)
            continue
        missing = missing_optional_dependencies(module)
        if missing:
            source_only.append(module)
            optional_skips.append({"module": module, "missing_dependencies": missing,
                                   "scope": "AST/import-closure only; preparation not executed"})
            continue
        importlib.import_module(module)
        imported.append(module)
    # Exercise the actual dynamic loader rather than merely testing file existence.
    loader = importlib.import_module("research.lifecycle_memory.run_phase2_mech_job1_instrument_gates")
    helper = loader._load_job0_runtime(root)
    require(callable(helper._apply_chat_template_direct), "job0 helper API missing")
    require(not HEAVY.intersection(sys.modules), "model backend loaded during smoke")
    return {"imported_scientific_modules": len(imported), "dynamic_job0_loaded": True,
            "source_only_modules": source_only, "optional_dependency_skips": optional_skips,
            "heavy_imports": [], "model_or_gpu_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="Requires NumPy and Pillow; reports optional SciPy skips; never a model backend")
    args = parser.parse_args()
    result = verify(upstream=args.upstream)
    if args.smoke:
        result["smoke"] = smoke()
    print(json.dumps(result, indent=2, sort_keys=True))
