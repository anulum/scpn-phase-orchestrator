# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Required genuine branch profile aggregation

"""Admit real coverage profiles and refuse unexplained fault-fixture credit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from tools.branch_profile_provenance import (
    FIXTURE_MARKER,
    PROFILES,
    ProfileReceipt,
    dump_json,
    license_artifact,
    prove_absence,
    record_profile,
    sha256,
    source_inputs,
    source_member,
)


def _require_untouched(directory: Path, receipt: ProfileReceipt) -> None:
    """Refuse a raw database whose bytes differ from its recorded identity."""
    if sha256(directory / receipt["database"]) != receipt["database_sha256"]:
        raise ValueError(f"{receipt['profile']}: raw database hash mismatch")


def _read_receipt(directory: Path, profile: str, revision: str) -> ProfileReceipt:
    """Validate raw mode, identity and database integrity before any combine."""
    value = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 2
        or value.get("profile") != profile
        or value.get("revision") != revision
    ):
        raise ValueError(f"{profile}: missing, stale or malformed profile receipt")
    receipt = cast(ProfileReceipt, value)
    for name in ("source_hashes", "input_hashes", "environment", "kernel", "fixture"):
        field = value.get(name)
        if not isinstance(field, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in field.items()
        ):
            raise ValueError(f"{profile}: malformed {name}")
    for name in ("package_root", "source_root", "database_sha256"):
        if not isinstance(value.get(name), str):
            raise ValueError(f"{profile}: malformed {name}")
    database_name = value.get("database")
    if not isinstance(database_name, str) or Path(database_name).name != database_name:
        raise ValueError(f"{profile}: unsafe or missing raw database name")
    database = directory / database_name
    _require_untouched(directory, receipt)
    raw = CoverageData(basename=str(database))
    raw.read()
    if not raw.has_arcs() or not raw.measured_files():
        raise ValueError(f"{profile}: missing, empty or statement-only database")
    package_root = Path(receipt["package_root"])
    source_root = Path(receipt["source_root"])
    for name in raw.measured_files():
        if (
            "scpn_phase_orchestrator/" + source_member(name, package_root, source_root)
            not in receipt["source_hashes"]
        ):
            raise ValueError(f"{profile}: unbound measured source")
    if profile.startswith("defective-"):
        if (
            receipt["fixture"].get("marker") != FIXTURE_MARKER
            or receipt["kernel"].get("marker") != FIXTURE_MARKER
            or not receipt["fixture"].get("wheel_sha256")
            or not receipt["fixture"].get("before_install_sha256")
            or not receipt["kernel"].get("sha256")
        ):
            raise ValueError(f"{profile}: missing genuine fault fixture provenance")
        expected_variant = {
            "defective-output": "invalid-outputs",
            "defective-missing": "missing-classes",
        }[profile]
        proof_name = receipt["fixture"].get("before_install", "")
        if (
            not proof_name
            or Path(proof_name).name != proof_name
            or receipt["kernel"].get("variant") != expected_variant
        ):
            raise ValueError(f"{profile}: invalid fixture variant or absence proof")
        proof_path = directory / proof_name
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if not isinstance(proof, dict) or not isinstance(proof.get("environment"), str):
            raise ValueError(f"{profile}: malformed absent-kernel observation")
        if (
            sha256(proof_path) != receipt["fixture"]["before_install_sha256"]
            or proof.get("kernel_importable") is not False
            or proof.get("genuine_distribution_installed") is not False
        ):
            raise ValueError(
                f"{profile}: absent-kernel observation is missing or changed"
            )
        Path(receipt["package_root"]).relative_to(proof["environment"])
        Path(receipt["kernel"]["extension"]).relative_to(proof["environment"])
    elif profile == "absent" and (receipt["kernel"] or receipt["fixture"]):
        raise ValueError("absent profile contains a kernel or fault fixture")
    elif profile == "native" and (
        not receipt["kernel"].get("sha256")
        or receipt["kernel"].get("marker")
        or receipt["fixture"]
    ):
        raise ValueError(
            "native profile lacks a genuine extension or contains a fixture"
        )
    return receipt


def _reviewed_residuals(root: Path) -> dict[str, object]:
    """Bind the approved guard list and unchanged ratchet to actual source bytes."""
    document = json.loads(
        (root / "tools/branch_profile_residuals.json").read_text(encoding="utf-8")
    )
    if document["thresholds_sha256"] != sha256(
        root / "tools/coverage_guard_branch_thresholds.json"
    ):
        raise ValueError("branch thresholds changed from the reviewed disposition")
    for filename, entry in document["files"].items():
        source = root / filename
        if sha256(source) != entry["source_sha256"]:
            raise ValueError(f"reviewed guard source changed: {filename}")
        lines = source.read_bytes().splitlines(keepends=True)
        for number, expected in entry["line_hashes"].items():
            if hashlib.sha256(lines[int(number) - 1]).hexdigest() != expected:
                raise ValueError(f"reviewed guard line changed: {filename}:{number}")
    return cast(dict[str, object], document)


def _combine(
    root: Path,
    destination: Path,
    name: str,
    profiles: list[tuple[Path, ProfileReceipt]],
    config: Path,
) -> tuple[Coverage, dict[str, object]]:
    """Combine untouched real databases and verify their exact line/arc union.

    The combined data must hold exactly the source members of the raw databases
    under their checkout paths, and for each member exactly the union of the raw
    arcs. A member measured both in the installed package and in the checkout
    source it was built from is one member. A checkout member that resolves to
    another path, such as a symbolic link, changes the membership and is refused.
    """
    expected: dict[str, set[tuple[int, int]]] = {}
    inputs: list[str] = []
    for directory, receipt in profiles:
        filename = directory / receipt["database"]
        inputs.append(str(filename))
        raw = CoverageData(basename=str(filename))
        raw.read()
        for measured in raw.measured_files():
            relative = source_member(
                measured, Path(receipt["package_root"]), Path(receipt["source_root"])
            )
            canonical = str(root / "src/scpn_phase_orchestrator" / relative)
            expected.setdefault(canonical, set()).update(raw.arcs(measured) or [])
    measurement = Coverage(
        config_file=str(config), data_file=str(destination / (".coverage." + name))
    )
    measurement.combine(data_paths=inputs, strict=True, keep=True)
    measurement.save()
    license_artifact(destination / (".coverage." + name))
    data = measurement.get_data()
    combined = {name: set(data.arcs(name) or []) for name in data.measured_files()}
    if combined != expected:
        raise ValueError(
            "combine did not preserve the exact raw source membership and arc union"
        )
    for directory, receipt in profiles:
        _require_untouched(directory, receipt)
    report_path = destination / (name + ".json")
    measurement.json_report(outfile=str(report_path))
    license_artifact(report_path)
    return measurement, cast(
        dict[str, object], json.loads(report_path.read_text(encoding="utf-8"))
    )


def combine_profiles(
    *, root: Path, directories: dict[str, Path], revision: str, destination: Path
) -> None:
    """Require all actual profiles, preserve genuine success coverage, then gate.

    The coverage reports name each source relative to the working directory,
    while the reviewed disposition and the ratchet name it relative to the
    checkout. The aggregate is therefore produced with the checkout as the
    working directory, wherever the caller stands. Relative arguments are taken
    from the caller's directory: every path is resolved before the working
    directory changes, and the caller's directory is restored afterwards.

    This is a command for one single-threaded process. The working directory is
    process-wide while the aggregate runs, and the coverage library keeps the
    checkout as its relative directory for the rest of the process.
    """
    if set(directories) != set(PROFILES):
        raise ValueError(
            "all four required native/absent/fault variant artifacts are obligatory"
        )
    root = root.resolve()
    destination = destination.resolve()
    resolved = {name: path.resolve() for name, path in directories.items()}
    with contextlib.chdir(root):
        _aggregate(
            root=root, directories=resolved, revision=revision, destination=destination
        )


def _aggregate(
    *, root: Path, directories: dict[str, Path], revision: str, destination: Path
) -> None:
    """Admit the four profiles of one checkout from inside that checkout."""
    receipts = {
        profile: _read_receipt(directories[profile], profile, revision)
        for profile in PROFILES
    }
    genuine = receipts["native"]
    expected_python = {
        path.relative_to(root / "src").as_posix()
        for path in (root / "src/scpn_phase_orchestrator").rglob("*.py")
    }
    if {
        name for name in genuine["source_hashes"] if name.endswith(".py")
    } != expected_python or genuine["input_hashes"] != source_inputs(root):
        raise ValueError("profile source or common input membership is incomplete")
    for profile, receipt in receipts.items():
        if (
            receipt["source_hashes"],
            receipt["input_hashes"],
            receipt["environment"],
        ) != (
            genuine["source_hashes"],
            genuine["input_hashes"],
            genuine["environment"],
        ):
            raise ValueError(
                f"{profile}: source, lock, configuration or environment mismatch"
            )
    for filename, expected in genuine["source_hashes"].items():
        if sha256(root / "src" / filename) != expected:
            raise ValueError(f"aggregate checkout source mismatch: {filename}")
    for receipt in (receipts["defective-output"], receipts["defective-missing"]):
        for filename in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "src/lib.rs"):
            if receipt["fixture"].get(filename) != sha256(
                root / "tests/native_output_fixture" / filename
            ):
                raise ValueError(
                    "fixture source or lock differs from the aggregate checkout"
                )
    residuals = _reviewed_residuals(root)
    if destination.exists():
        raise ValueError(
            "aggregate destination already exists; preserve previous evidence"
        )
    destination.mkdir(parents=True)
    config = destination / "measurement-config.toml"
    aliases = list(
        dict.fromkeys(
            [str(root / "src/scpn_phase_orchestrator")]
            + [receipts[profile]["package_root"] for profile in PROFILES]
            + [receipts[profile]["source_root"] for profile in PROFILES]
        )
    )
    config.write_text(
        (root / "pyproject.toml").read_text(encoding="utf-8")
        + "\n[tool.coverage.paths]\nprofile_package = "
        + json.dumps(aliases)
        + "\n",
        encoding="utf-8",
    )
    baseline, before = _combine(
        root,
        destination,
        "genuine",
        [(directories[p], receipts[p]) for p in ("native", "absent")],
        config,
    )
    complete, after = _combine(
        root,
        destination,
        "all",
        [(directories[p], receipts[p]) for p in PROFILES],
        config,
    )
    before_files = cast(dict[str, dict[str, object]], before["files"])
    after_files = cast(dict[str, dict[str, object]], after["files"])
    approved = cast(dict[str, dict[str, object]], residuals["files"])
    for filename, final in after_files.items():
        initial = before_files.get(filename)
        if initial is None:
            raise ValueError(
                f"fixture would introduce previously unmeasured source: {filename}"
            )
        new_lines = set(cast(list[int], final["executed_lines"])) - set(
            cast(list[int], initial["executed_lines"])
        )
        new_branches = {
            tuple(v) for v in cast(list[list[int]], final["executed_branches"])
        } - {tuple(v) for v in cast(list[list[int]], initial["executed_branches"])}
        entry = approved.get(filename, {"lines": [], "branches": []})
        if not new_lines <= set(
            cast(list[int], entry["lines"])
        ) or not new_branches <= {
            tuple(v) for v in cast(list[list[int]], entry["branches"])
        }:
            raise ValueError(
                f"fixture would hide an unreviewed coverage gap: {filename}"
            )
        if filename in approved:
            # The next two checks only name the cause more precisely. A gap of
            # the genuine profiles is already refused above when a fixture
            # profile executes it, and below when no profile does.
            if not set(cast(list[int], initial["missing_lines"])) <= set(
                cast(list[int], entry["lines"])
            ):
                raise ValueError(
                    f"genuine profiles leave an unreviewed success-path gap: {filename}"
                )
            if not {
                tuple(v) for v in cast(list[list[int]], initial["missing_branches"])
            } <= {tuple(v) for v in cast(list[list[int]], entry["branches"])}:
                raise ValueError(
                    f"genuine profiles leave an unreviewed branch gap: {filename}"
                )
            if (
                final["missing_lines"]
                or final["missing_branches"]
                or final["excluded_lines"]
            ):
                raise ValueError(
                    "three-profile union does not achieve "
                    f"zero-exclusion full coverage: {filename}"
                )
    if not set(approved) <= set(after_files):
        raise ValueError("aggregate is missing a reviewed owning source")
    complete.xml_report(outfile=str(destination / "coverage-branch.xml"))
    license_artifact(destination / "coverage-branch.xml")
    dump_json(
        destination / "admission.json",
        {
            "revision": revision,
            "profiles": list(PROFILES),
            "originals_unchanged": True,
            "exact_arc_union_verified": True,
            "guard_coverage": "100 % with three real profiles, 0 exclusions",
            "scope": (
                "reviewed E/I, sheaf and sparse guards; "
                "global ratchet is a separate required gate"
            ),
            "residuals_sha256": sha256(root / "tools/branch_profile_residuals.json"),
            "genuine_data": str(destination / ".coverage.genuine"),
            "genuine_data_sha256": sha256(destination / ".coverage.genuine"),
            "all_data": str(destination / ".coverage.all"),
            "all_data_sha256": sha256(destination / ".coverage.all"),
        },
    )


def main(arguments: list[str] | None = None) -> int:
    """Expose producer observation and fail-closed aggregation through one CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    absent = commands.add_parser("prove-absent")
    absent.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record")
    record.add_argument("--root", type=Path, default=Path.cwd())
    record.add_argument("--profile", choices=PROFILES, required=True)
    record.add_argument("--revision", required=True)
    record.add_argument("--database", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--fixture-wheel", type=Path)
    record.add_argument("--before-install", type=Path)
    combine = commands.add_parser("combine")
    combine.add_argument("--root", type=Path, default=Path.cwd())
    combine.add_argument("--revision", required=True)
    combine.add_argument("--destination", type=Path, required=True)
    for profile in PROFILES:
        combine.add_argument("--" + profile, type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        if args.command == "prove-absent":
            prove_absence(args.output)
        elif args.command == "record":
            record_profile(
                root=args.root.resolve(),
                profile=args.profile,
                revision=args.revision,
                database=args.database.resolve(),
                output=args.output,
                fixture_wheel=args.fixture_wheel,
                before_install=args.before_install,
            )
        else:
            combine_profiles(
                root=args.root.resolve(),
                revision=args.revision,
                destination=args.destination,
                directories={
                    profile: getattr(args, profile.replace("-", "_"))
                    for profile in PROFILES
                },
            )
    except (ValueError, OSError, KeyError, CoverageException) as error:
        print(f"branch profile refusal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
