# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Branch profile runtime provenance

"""Bind actual installed runtimes and measured databases to source identity."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import TypedDict

from coverage import CoverageData

PROFILES = ("native", "absent", "defective-output", "defective-missing")
FIXTURE_MARKER = "SPO_NATIVE_OUTPUT_FAULT_FIXTURE_V1"
_ARTIFACT_LICENSE = (
    "\n".join(Path(__file__).read_text(encoding="utf-8").splitlines()[:7])
    + "\n# SPDX-FileCopyrightText: Concepts 1996–2026 Miroslav Šotek\n"
    + "# SPDX-FileCopyrightText: Code 2020–2026 Miroslav Šotek\n"
)


class ProfileReceipt(TypedDict):
    """Identify the source, actual installed runtime and untouched branch data."""

    schema_version: int
    profile: str
    revision: str
    source_hashes: dict[str, str]
    input_hashes: dict[str, str]
    environment: dict[str, str]
    package_root: str
    source_root: str
    database: str
    database_sha256: str
    kernel: dict[str, str]
    fixture: dict[str, str]


def sha256(path: Path) -> str:
    """Hash actual file bytes without normalising source or coverage data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def license_artifact(path: Path) -> None:
    """Retain valid machine formats and attach ownership without rewriting bytes."""
    sidecar = path.with_name(path.name + ".license")
    if not sidecar.exists():
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(_ARTIFACT_LICENSE)


def dump_json(path: Path, value: object) -> None:
    """Write a new JSON artifact without replacing any previous receipt."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    license_artifact(path)


def prove_absence(output: Path) -> None:
    """Observe actual absence before installing the defective native fixture."""
    if importlib.util.find_spec("spo_kernel") is not None:
        raise ValueError(
            "spo_kernel is already importable; refuse fixture installation"
        )
    try:
        importlib.metadata.distribution("spo-kernel")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("genuine spo-kernel distribution is already installed")
    dump_json(
        output,
        {
            "kernel_importable": False,
            "genuine_distribution_installed": False,
            "executable": sys.executable,
            "environment": str(Path(sys.prefix).resolve()),
        },
    )


def _installed_sources(root: Path) -> tuple[Path, dict[str, str]]:
    """Require one whole installed package matching every packaged source member."""
    package = importlib.import_module("scpn_phase_orchestrator")
    if package.__file__ is None:
        raise ValueError("installed package is an incomplete namespace")
    package_root = Path(package.__file__).resolve().parent
    package_root.relative_to(Path(sys.prefix).resolve())
    distribution = importlib.metadata.distribution("scpn-phase-orchestrator")
    sources: dict[str, str] = {}
    for member in distribution.files or []:
        relative = Path(str(member))
        if relative.parts[0] != "scpn_phase_orchestrator" or relative.suffix == ".pyc":
            continue
        source = root / "src" / relative
        installed = Path(str(distribution.locate_file(member)))
        if not source.is_file() or sha256(source) != sha256(installed):
            raise ValueError(f"installed package source mismatch: {relative}")
        sources[relative.as_posix()] = sha256(source)
    expected = {
        path.relative_to(root / "src").as_posix()
        for path in (root / "src/scpn_phase_orchestrator").rglob("*.py")
    }
    measured_python = {name for name in sources if name.endswith(".py")}
    if measured_python != expected or not sources:
        raise ValueError(
            "installed package has incomplete or unexpected Python membership"
        )
    return package_root, sources


def source_member(measured: str, package_root: Path, source_root: Path) -> str:
    """Name a measured file by its member path inside the package.

    A measured file is a member of the installed package or the same member in
    the checkout that the package was built from. Tests start child
    interpreters on the checkout source where no installed package can exist,
    for example in a fresh environment before the kernel wheel is installed.
    A profile is recorded only after every checkout member has been found equal
    to its installed copy byte for byte, and the aggregate compares the
    checkout with the recorded hashes again, so both names identify one source.
    Anything else is refused.
    """
    path = Path(measured)
    for base in (package_root, source_root):
        if path.is_relative_to(base):
            return path.relative_to(base).as_posix()
    raise ValueError(
        f"measured file is outside the installed package and its source: {measured}"
    )


def _kernel_identity(profile: str) -> dict[str, str]:
    """Verify actual absence, a genuine extension, or the marked fault extension."""
    spec = importlib.util.find_spec("spo_kernel")
    if profile == "absent":
        if spec is not None:
            raise ValueError("absent profile has an importable spo_kernel")
        try:
            importlib.metadata.distribution("spo-kernel")
        except importlib.metadata.PackageNotFoundError:
            return {}
        raise ValueError("absent profile retains genuine kernel distribution metadata")
    if spec is None:
        raise ValueError(f"{profile} profile has no installed extension")
    module = importlib.import_module("spo_kernel")
    extension = module
    if not any(
        str(module.__file__).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES
    ):
        extension = importlib.import_module("spo_kernel.spo_kernel")
    if extension.__file__ is None:
        raise ValueError("kernel producer has no extension file")
    path = Path(extension.__file__).resolve()
    path.relative_to(Path(sys.prefix).resolve())
    if not any(str(path).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
        raise ValueError("kernel producer is not an actual compiled extension")
    marker = getattr(module, "__SPO_FAULT_FIXTURE__", "")
    if profile == "native":
        if marker:
            raise ValueError("native profile contains a fault fixture")
        version = importlib.metadata.version("spo-kernel")
    else:
        variant = {
            "defective-output": "invalid-outputs",
            "defective-missing": "missing-classes",
        }[profile]
        if (
            marker != FIXTURE_MARKER
            or getattr(module, "__SPO_FIXTURE_VARIANT__", "") != variant
        ):
            raise ValueError(
                "defective profile has a missing or mismatched fixture marker"
            )
        try:
            importlib.metadata.distribution("spo-kernel")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("defective profile also has a genuine kernel installed")
        version = importlib.metadata.version("spo-native-output-fixture")
    return {
        "extension": str(path),
        "sha256": sha256(path),
        "version": version,
        "marker": str(marker),
        "variant": str(getattr(module, "__SPO_FIXTURE_VARIANT__", "")),
    }


def source_inputs(root: Path) -> dict[str, str]:
    """Hash the complete common lock, configuration and native source inputs."""
    inputs = {
        name: sha256(root / name)
        for name in (
            "pyproject.toml",
            "requirements/dev-lock.txt",
            "requirements/ci-tools.txt",
            "requirements/build-tools.txt",
            "requirements/studio-sdk.txt",
            "spo-kernel/Cargo.lock",
            "spo-kernel/Cargo.toml",
            "tools/coverage_guard_branch_thresholds.json",
            "tools/branch_profile_provenance.py",
            "tools/branch_coverage_profiles.py",
            "tools/branch_profile_residuals.json",
        )
    }
    for path in (root / "spo-kernel/crates").rglob("*"):
        if path.is_file() and path.suffix in {".rs", ".toml"}:
            inputs[path.relative_to(root).as_posix()] = sha256(path)
    return inputs


def record_profile(
    *,
    root: Path,
    profile: str,
    revision: str,
    database: Path,
    output: Path,
    fixture_wheel: Path | None = None,
    before_install: Path | None = None,
) -> None:
    """Record an actual whole-package runtime and an existing branch database."""
    if profile not in PROFILES:
        raise ValueError(f"unknown required branch profile: {profile}")
    package_root, sources = _installed_sources(root)
    raw = CoverageData(basename=str(database))
    raw.read()
    if not raw.has_arcs() or not raw.measured_files():
        raise ValueError("profile database is missing, empty or statement-only")
    source_root = root / "src/scpn_phase_orchestrator"
    for name in raw.measured_files():
        key = "scpn_phase_orchestrator/" + source_member(
            name, package_root, source_root
        )
        if key not in sources:
            raise ValueError(f"database measured unbound source: {name}")
    inputs = source_inputs(root)
    kernel = _kernel_identity(profile)
    fixture: dict[str, str] = {}
    if profile.startswith("defective-"):
        if fixture_wheel is None or before_install is None:
            raise ValueError(
                "defective profile lacks its wheel or pre-install absence receipt"
            )
        proof = json.loads(before_install.read_text(encoding="utf-8"))
        if (
            proof.get("kernel_importable") is not False
            or proof.get("genuine_distribution_installed") is not False
            or proof.get("environment") != str(Path(sys.prefix).resolve())
        ):
            raise ValueError(
                "fixture pre-install absence proof is invalid "
                "or from another environment"
            )
        fixture = {
            "wheel_sha256": sha256(fixture_wheel),
            "before_install_sha256": sha256(before_install),
            "marker": FIXTURE_MARKER,
        }
        retained_proof = output.parent / "before-install.json"
        with retained_proof.open("xb") as stream:
            stream.write(before_install.read_bytes())
        license_artifact(retained_proof)
        fixture["before_install"] = retained_proof.name
        for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "src/lib.rs"):
            fixture[name] = sha256(root / "tests/native_output_fixture" / name)
    receipt: ProfileReceipt = {
        "schema_version": 2,
        "profile": profile,
        "revision": revision,
        "source_hashes": sources,
        "input_hashes": inputs,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("coverage", "numpy", "scipy")
            },
        },
        "package_root": str(package_root),
        "source_root": str(source_root),
        "database": database.name,
        "database_sha256": sha256(database),
        "kernel": kernel,
        "fixture": fixture,
    }
    license_artifact(database)
    dump_json(output, receipt)
