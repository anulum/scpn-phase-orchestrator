#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator capability manifest generator

"""Generate a deterministic public capability inventory from repository files.

The manifest is static-source inventory only. It centralises public counts for
README, documentation, and release checks without importing optional runtime
dependencies or making benchmark, coverage, hardware, or scientific-fidelity
claims.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python 3.10 fallback path.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


CAPABILITY_MANIFEST_SCHEMA_VERSION = "capability-manifest.v1"
DEFAULT_JSON_OUTPUT = Path("docs/_generated/capability_manifest.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/_generated/capability_snapshot.md")
DEFAULT_CONFIG = Path("tools/capability_manifest.toml")
DEFAULT_README = Path("README.md")
DEFAULT_MARKER_START = "<!-- capability-snapshot:start -->"
DEFAULT_MARKER_END = "<!-- capability-snapshot:end -->"


def _default_labels() -> dict[str, str]:
    return {
        "version": "Package version",
        "public_api_exports": "Public API exports",
        "python_package_modules": "Python package modules",
        "core_engine_modules": "Core Engine modules",
        "runtime_serving_modules": "Runtime/Serving modules",
        "integration_modules": "Integration modules",
        "research_experimental_modules": "Research/Experimental modules",
        "domainpacks": "Domainpack files",
        "rust_kernel_files": "Rust kernel files",
        "optional_extras": "Optional extras",
        "test_files": "Python test files",
        "public_documentation_pages": "Public documentation pages",
        "github_workflows": "GitHub Actions workflows",
    }


@dataclass(frozen=True)
class CapabilityManifestConfig:
    """Portable configuration for repository capability inventory."""

    project_label: str
    schema_version: str
    json_output: Path
    markdown_output: Path
    readme_path: Path
    readme_marker_start: str
    readme_marker_end: str
    package_root: Path
    domainpacks_root: Path
    tests_root: Path
    docs_root: Path
    workflows_root: Path
    rust_root: Path
    exclude_doc_parts: tuple[str, ...]
    labels: dict[str, str]
    product_boundaries: dict[str, tuple[str, ...]]
    source_path: Path | None


def load_config(
    repo: Path, config_path: Path | None = None
) -> CapabilityManifestConfig:
    """Load capability-manifest configuration from TOML."""

    repo = repo.resolve()
    path = repo / (config_path or DEFAULT_CONFIG)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    readme = raw.get("readme", {})
    labels = _default_labels()
    labels.update(
        {str(key): str(value) for key, value in raw.get("labels", {}).items()}
    )
    product_boundaries = {
        str(name): tuple(str(part) for part in parts)
        for name, parts in raw.get("product_boundaries", {}).items()
    }
    return CapabilityManifestConfig(
        project_label=str(raw.get("project_label", "SCPN Phase Orchestrator")),
        schema_version=str(
            raw.get("schema_version", CAPABILITY_MANIFEST_SCHEMA_VERSION)
        ),
        json_output=Path(paths.get("json_output", DEFAULT_JSON_OUTPUT.as_posix())),
        markdown_output=Path(
            paths.get("markdown_output", DEFAULT_MARKDOWN_OUTPUT.as_posix())
        ),
        readme_path=Path(readme.get("path", DEFAULT_README.as_posix())),
        readme_marker_start=str(readme.get("marker_start", DEFAULT_MARKER_START)),
        readme_marker_end=str(readme.get("marker_end", DEFAULT_MARKER_END)),
        package_root=Path(paths.get("package_root", "src/scpn_phase_orchestrator")),
        domainpacks_root=Path(paths.get("domainpacks_root", "domainpacks")),
        tests_root=Path(paths.get("tests_root", "tests")),
        docs_root=Path(paths.get("docs_root", "docs")),
        workflows_root=Path(paths.get("workflows_root", ".github/workflows")),
        rust_root=Path(paths.get("rust_root", "spo-kernel")),
        exclude_doc_parts=tuple(
            str(part)
            for part in raw.get("exclude_doc_parts", ["internal", "_generated"])
        ),
        labels=labels,
        product_boundaries=product_boundaries,
        source_path=_relative_path(path, repo) if path.exists() else None,
    )


def build_capability_manifest(
    repo: Path, config: CapabilityManifestConfig | None = None
) -> dict[str, Any]:
    """Build a deterministic capability manifest for public surfaces."""

    repo = repo.resolve()
    config = config or load_config(repo)
    pyproject = _load_pyproject(repo / "pyproject.toml")
    package_modules = _python_files(repo / config.package_root, repo=repo)
    product_boundaries = _product_boundary_modules(repo, config)
    public_exports = _public_exports(repo / config.package_root / "__init__.py")
    domainpacks = _domainpack_files(repo / config.domainpacks_root, repo=repo)
    rust_files = _rust_files(repo / config.rust_root, repo=repo)
    tests = _python_files(repo / config.tests_root, repo=repo)
    docs_pages = _markdown_docs(
        repo / config.docs_root,
        repo=repo,
        exclude_parts=config.exclude_doc_parts,
    )
    workflows = _workflow_files(repo / config.workflows_root, repo=repo)
    extras = _project_extras(pyproject)
    package_data_key = config.package_root.name

    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": config.schema_version,
        "project_label": config.project_label,
        "generated_from": {
            "config": str(config.source_path)
            if config.source_path is not None
            else "built-in defaults",
            "generator": "tools/capability_manifest.py",
        },
        "project": {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "requires_python": pyproject["project"]["requires-python"],
            "readme": pyproject["project"]["readme"],
            "license": pyproject["project"]["license"],
        },
        "labels": config.labels,
        "counts": {
            "public_api_exports": len(public_exports),
            "python_package_modules": len(package_modules),
            "core_engine_modules": len(product_boundaries.get("core_engine", [])),
            "runtime_serving_modules": len(
                product_boundaries.get("runtime_serving", [])
            ),
            "integration_modules": len(product_boundaries.get("integrations", [])),
            "research_experimental_modules": len(
                product_boundaries.get("research_experimental", [])
            ),
            "domainpacks": len(domainpacks),
            "rust_kernel_files": len(rust_files),
            "optional_extras": len(extras),
            "test_files": len(tests),
            "public_documentation_pages": len(docs_pages),
            "github_workflows": len(workflows),
        },
        "package_exports": public_exports,
        "python_package_modules": package_modules,
        "product_boundaries": product_boundaries,
        "domainpacks": domainpacks,
        "rust_kernel_files": rust_files,
        "packaging": {
            "optional_extras": extras,
            "shipped_package_data": pyproject.get("tool", {})
            .get("setuptools", {})
            .get("package-data", {})
            .get(package_data_key, []),
        },
        "quality_gates": {
            "test_files": tests,
            "github_workflows": workflows,
        },
        "documentation": {
            "public_pages": docs_pages,
        },
        "evidence_boundary": (
            "Counts are file-system and static-source inventory only; benchmark, "
            "coverage, hardware, and scientific-fidelity claims remain governed by "
            "their dedicated evidence artifacts."
        ),
    }


def render_markdown_snapshot(manifest: dict[str, Any]) -> str:
    """Render a README-safe Markdown snapshot from a manifest."""

    counts = manifest["counts"]
    project = manifest["project"]
    labels = manifest.get("labels", _default_labels())
    rows = [
        (labels["version"], project["version"]),
        (labels["public_api_exports"], counts["public_api_exports"]),
        (labels["python_package_modules"], counts["python_package_modules"]),
        (labels["core_engine_modules"], counts["core_engine_modules"]),
        (labels["runtime_serving_modules"], counts["runtime_serving_modules"]),
        (labels["integration_modules"], counts["integration_modules"]),
        (
            labels["research_experimental_modules"],
            counts["research_experimental_modules"],
        ),
        (labels["domainpacks"], counts["domainpacks"]),
        (labels["rust_kernel_files"], counts["rust_kernel_files"]),
        (labels["optional_extras"], counts["optional_extras"]),
        (labels["test_files"], counts["test_files"]),
        (labels["public_documentation_pages"], counts["public_documentation_pages"]),
        (labels["github_workflows"], counts["github_workflows"]),
    ]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Generated by tools/capability_manifest.py; "
        "do not edit counts by hand. -->",
        "",
        f"### {manifest.get('project_label', 'Project')} Capability Inventory",
        "",
        "| Surface | Current inventory |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.extend(
        [
            "",
            (
                "Evidence boundary: this snapshot is a static inventory. Performance, "
                "coverage, hardware, and scientific-fidelity claims require their own "
                "committed evidence artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def refresh_readme_block(
    repo: Path,
    snapshot: str,
    *,
    config: CapabilityManifestConfig,
) -> Path:
    """Refresh the README block bounded by configured markers."""

    readme_path = repo / config.readme_path
    text = readme_path.read_text(encoding="utf-8")
    start = config.readme_marker_start
    end = config.readme_marker_end
    if start not in text or end not in text:
        raise RuntimeError(
            f"{config.readme_path} is missing capability snapshot markers"
        )
    before, rest = text.split(start, maxsplit=1)
    _old, after = rest.split(end, maxsplit=1)
    readme_path.write_text(
        before + f"{start}\n{snapshot.rstrip()}\n{end}" + after,
        encoding="utf-8",
    )
    return readme_path


def write_outputs(
    manifest: dict[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown outputs."""

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown_snapshot(manifest), encoding="utf-8")
    return json_output, markdown_output


def refresh_outputs(
    repo: Path,
    *,
    config: CapabilityManifestConfig,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    update_readme: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Regenerate JSON, Markdown, and optionally the README snapshot."""

    manifest = build_capability_manifest(repo, config)
    json_path, markdown_path = write_outputs(
        manifest,
        json_output=repo / (json_output or config.json_output),
        markdown_output=repo / (markdown_output or config.markdown_output),
    )
    readme_path = None
    if update_readme:
        readme_path = refresh_readme_block(
            repo,
            render_markdown_snapshot(manifest),
            config=config,
        )
    return json_path, markdown_path, readme_path


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest structure and count/list consistency."""

    errors: list[str] = []
    if payload.get("schema_version") != CAPABILITY_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    for key in (
        "project",
        "counts",
        "package_exports",
        "python_package_modules",
        "product_boundaries",
        "domainpacks",
        "rust_kernel_files",
        "packaging",
        "quality_gates",
        "documentation",
    ):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
    else:
        for key, value in counts.items():
            if not isinstance(value, int) or value < 0:
                errors.append(f"counts.{key} must be a non-negative integer")
        _check_count(
            errors,
            counts,
            "public_api_exports",
            payload.get("package_exports"),
        )
        _check_count(
            errors,
            counts,
            "python_package_modules",
            payload.get("python_package_modules"),
        )
        boundaries = payload.get("product_boundaries", {})
        if isinstance(boundaries, dict):
            _check_count(
                errors,
                counts,
                "core_engine_modules",
                boundaries.get("core_engine"),
            )
            _check_count(
                errors,
                counts,
                "runtime_serving_modules",
                boundaries.get("runtime_serving"),
            )
            _check_count(
                errors,
                counts,
                "integration_modules",
                boundaries.get("integrations"),
            )
            _check_count(
                errors,
                counts,
                "research_experimental_modules",
                boundaries.get("research_experimental"),
            )
        else:
            errors.append("product_boundaries must be an object")
        _check_count(errors, counts, "domainpacks", payload.get("domainpacks"))
        _check_count(
            errors,
            counts,
            "rust_kernel_files",
            payload.get("rust_kernel_files"),
        )
        packaging = payload.get("packaging", {})
        if isinstance(packaging, dict):
            _check_count(
                errors,
                counts,
                "optional_extras",
                packaging.get("optional_extras"),
            )
        quality = payload.get("quality_gates", {})
        if isinstance(quality, dict):
            _check_count(errors, counts, "test_files", quality.get("test_files"))
            _check_count(
                errors,
                counts,
                "github_workflows",
                quality.get("github_workflows"),
            )
        documentation = payload.get("documentation", {})
        if isinstance(documentation, dict):
            _check_count(
                errors,
                counts,
                "public_documentation_pages",
                documentation.get("public_pages"),
            )
    return {"passed": not errors, "errors": errors}


def assert_outputs_current(
    repo: Path,
    *,
    config: CapabilityManifestConfig | None = None,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    check_readme: bool = True,
) -> None:
    """Raise if generated outputs drift from current sources."""

    config = config or load_config(repo)
    manifest = build_capability_manifest(repo, config)
    expected_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    expected_markdown = render_markdown_snapshot(manifest)
    json_path = repo / (json_output or config.json_output)
    markdown_path = repo / (markdown_output or config.markdown_output)
    errors: list[str] = []
    if not json_path.exists():
        errors.append(f"missing generated manifest: {json_path.relative_to(repo)}")
    elif json_path.read_text(encoding="utf-8") != expected_json:
        errors.append(f"stale generated manifest: {json_path.relative_to(repo)}")
    if not markdown_path.exists():
        errors.append(f"missing generated snapshot: {markdown_path.relative_to(repo)}")
    elif markdown_path.read_text(encoding="utf-8") != expected_markdown:
        errors.append(f"stale generated snapshot: {markdown_path.relative_to(repo)}")
    if check_readme and not _readme_block_matches(
        repo / config.readme_path, expected_markdown, config=config
    ):
        errors.append(f"stale README capability block: {config.readme_path}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _check_count(
    errors: list[str],
    counts: dict[str, Any],
    key: str,
    values: Any,
) -> None:
    if not isinstance(values, list):
        errors.append(f"list missing for count {key}")
        return
    if counts.get(key) != len(values):
        errors.append(f"counts.{key} does not match list length")


def _load_pyproject(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _public_exports(init_path: Path) -> list[str]:
    if not init_path.exists():
        return []
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return sorted(_literal_string_list(node.value))
    return []


def _literal_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


def _project_extras(pyproject: dict[str, Any]) -> list[str]:
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    if not isinstance(extras, dict):
        return []
    return sorted(str(name) for name in extras)


def _domainpack_files(root: Path, *, repo: Path) -> list[str]:
    if not root.exists():
        return []
    return [_rel(path, repo) for path in sorted(root.rglob("binding_spec.yaml"))]


def _product_boundary_modules(
    repo: Path,
    config: CapabilityManifestConfig,
) -> dict[str, list[str]]:
    package_root = repo / config.package_root
    boundaries: dict[str, list[str]] = {}
    for name, parts in sorted(config.product_boundaries.items()):
        modules: list[str] = []
        for part in parts:
            path = package_root / part
            if path.is_dir():
                modules.extend(_python_files(path, repo=repo))
            elif path.is_file() and path.suffix == ".py":
                modules.append(_rel(path, repo))
        boundaries[name] = sorted(set(modules))
    return boundaries


def _rust_files(root: Path, *, repo: Path) -> list[str]:
    if not root.exists():
        return []
    files = list(root.rglob("*.rs")) + list(root.rglob("Cargo.toml"))
    return [_rel(path, repo) for path in sorted(files)]


def _workflow_files(root: Path, *, repo: Path) -> list[str]:
    if not root.exists():
        return []
    files = list(root.glob("*.yml")) + list(root.glob("*.yaml"))
    return [_rel(path, repo) for path in sorted(files)]


def _python_files(root: Path, *, repo: Path) -> list[str]:
    if not root.exists():
        return []
    return [_rel(path, repo) for path in sorted(root.rglob("*.py"))]


def _markdown_docs(
    root: Path,
    *,
    repo: Path,
    exclude_parts: tuple[str, ...],
) -> list[str]:
    if not root.exists():
        return []
    return [
        _rel(path, repo)
        for path in sorted(root.rglob("*.md"))
        if not set(path.relative_to(root).parts).intersection(exclude_parts)
    ]


def _relative_path(path: Path, repo: Path) -> Path:
    try:
        return path.resolve().relative_to(repo)
    except ValueError:
        return path.resolve()


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _readme_block_matches(
    readme_path: Path,
    expected_markdown: str,
    *,
    config: CapabilityManifestConfig,
) -> bool:
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    start = config.readme_marker_start
    end = config.readme_marker_end
    if start not in text or end not in text:
        return False
    block = text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()
    return block == expected_markdown.strip()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = args.repo.resolve()
    config = load_config(repo, args.config)
    if args.validate is not None:
        report = validate_manifest(
            json.loads(args.validate.read_text(encoding="utf-8"))
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.check:
        try:
            assert_outputs_current(
                repo,
                config=config,
                json_output=args.output,
                markdown_output=args.markdown_output,
                check_readme=not args.no_readme,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    json_path, markdown_path, readme_path = refresh_outputs(
        repo,
        config=config,
        json_output=args.output,
        markdown_output=args.markdown_output,
        update_readme=not args.no_readme,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if readme_path is not None:
        print(f"Refreshed {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
