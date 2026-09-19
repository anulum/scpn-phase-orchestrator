# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Producer registry public contract tests

"""Exercise exact packaged ownership bytes without allocating direct ingress."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import coverage
import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics.producer_registry import (
    MAX_REACTOR_PRODUCER_REGISTRY_BYTES,
    REACTOR_PRODUCER_REGISTRY_SHA256,
    REACTOR_PRODUCER_REGISTRY_VERSION,
    reactor_producer_identity,
    reactor_producer_registry_bytes,
    reactor_producer_registry_from_bytes,
    reactor_producer_registry_schema,
)
from scpn_phase_orchestrator.reactor_semantics.semantic_profiles import (
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
)

RELEASE = {
    "version": REACTOR_PRODUCER_REGISTRY_VERSION,
    "digest": REACTOR_PRODUCER_REGISTRY_SHA256,
}


def test_packaged_registry_custody_and_all_configuration_queries() -> None:
    """Verify immutable ownership entries against every semantic profile."""
    data = reactor_producer_registry_bytes(**RELEASE)
    assert hashlib.sha256(data).hexdigest() == REACTOR_PRODUCER_REGISTRY_SHA256
    record = reactor_producer_registry_from_bytes(data, **RELEASE)
    profiles = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
    assert record["semantic_profile_registry_digest"] == profiles.digest
    assert record["reactor_registry_digest"] == profiles.reactor_registry_digest
    assert record["assignment_map_sha256"] == profiles.assignment_map_sha256
    assert record["authority"] == "review_only"
    assert record["actionable"] is False
    entries = cast("list[dict[str, object]]", record["entries"])
    assert [entry["configuration"] for entry in entries] == sorted(profiles.profiles)
    assert len(entries) == 34
    assert len({entry["device_project"] for entry in entries}) == 23
    for configuration, profile in profiles.profiles.items():
        entry = reactor_producer_identity(configuration, **RELEASE)
        assert entry["device_project"] == profile.device_project
        assert entry["direct_allocations"] == []
        assert entry["direct_ingress_state"] == "not_declared"
        historical = profile.to_record() if profile.producer_project else None
        assert entry["historical_review_adapter"] == historical
    record.clear()
    entry = reactor_producer_identity("conventional_tokamak", **RELEASE)
    cast("list[object]", entry["direct_allocations"]).append("invented")
    assert (
        reactor_producer_identity("conventional_tokamak", **RELEASE)[
            "direct_allocations"
        ]
        == []
    )
    assert reactor_producer_registry_bytes(**RELEASE) == data


@pytest.mark.parametrize(
    "version,digest",
    [("9.0.0", REACTOR_PRODUCER_REGISTRY_SHA256), ("1.0.0", "0" * 64)],
)
def test_unknown_or_changed_release_is_not_substituted(
    version: str, digest: str
) -> None:
    """Reject unknown registry pins without falling back to a shipped release."""
    with pytest.raises(ValueError, match="unrecognised"):
        reactor_producer_registry_bytes(version=version, digest=digest)
    with pytest.raises(ValueError, match="unrecognised"):
        reactor_producer_identity(
            "conventional_tokamak", version=version, digest=digest
        )


@pytest.mark.parametrize("configuration", ["tokamak", "unknown", "SCPN-CONTROL"])
def test_noncanonical_configuration_queries_are_refused(configuration: str) -> None:
    """Reject aliases and unknown names in exact configuration lookup."""
    with pytest.raises(ValueError, match="configuration is not"):
        reactor_producer_identity(configuration, **RELEASE)


def test_byte_decoder_refuses_noncanonical_and_untrusted_json_before_parsing() -> None:
    """Reject bytes outside the exact packaged snapshot and its size bound."""
    data = reactor_producer_registry_bytes(**RELEASE)
    candidates = (
        b"",
        data.rstrip(),
        json.dumps(json.loads(data), indent=2).encode(),
        b'{"entries":[],"entries":[]}',
        b'{"entries":NaN}',
        b"[" * 2000,
        b"\xff",
    )
    for candidate in candidates:
        with pytest.raises(ValueError, match="do not match"):
            reactor_producer_registry_from_bytes(candidate, **RELEASE)
    with pytest.raises(ValueError, match="bounded bytes"):
        reactor_producer_registry_from_bytes(
            b" " * (MAX_REACTOR_PRODUCER_REGISTRY_BYTES + 1), **RELEASE
        )


def test_generated_schema_and_runtime_refuse_role_and_owner_mutations() -> None:
    """Refuse invented owners and direct allocations in schema and byte ingress."""
    schema = reactor_producer_registry_schema(**RELEASE)
    assert (
        json.loads(
            Path("docs/specs/reactor_producer_registry.schema.json").read_bytes()
        )
        == schema
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    data = reactor_producer_registry_bytes(**RELEASE)
    validator.validate(reactor_producer_registry_from_bytes(data, **RELEASE))
    for index in range(34):
        for key, value in (
            ("device_project", "SCPN-CONTROL"),
            ("direct_ingress_state", "admitted"),
            ("direct_allocations", [{"role": "physical_observable"}]),
        ):
            record = reactor_producer_registry_from_bytes(data, **RELEASE)
            cast("list[dict[str, object]]", record["entries"])[index][key] = value
            assert not validator.is_valid(record)
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            with pytest.raises(ValueError, match="do not match"):
                reactor_producer_registry_from_bytes(encoded, **RELEASE)
            with pytest.raises(ValueError, match="unrecognised"):
                reactor_producer_registry_from_bytes(
                    encoded,
                    version=RELEASE["version"],
                    digest=hashlib.sha256(encoded).hexdigest(),
                )


def test_corrupted_package_data_is_refused_in_isolated_public_import(
    tmp_path: Path,
) -> None:
    """Refuse a corrupted packaged resource in a separate interpreter."""
    package = "scpn_phase_orchestrator"
    shutil.copytree(
        Path(__file__).parents[1] / "src" / package,
        tmp_path / package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    resource = tmp_path / package / "reactor_semantics/data/producer_registry_v1.json"
    resource.write_bytes(resource.read_bytes() + b" ")
    parent_coverage = coverage.Coverage.current()
    child_data = tmp_path / "registry.coverage"
    branch = bool(parent_coverage and parent_coverage.get_option("run:branch"))
    script = """
import sys
sys.path.insert(0, sys.argv[1])
child_coverage = None
if sys.argv[2]:
    import coverage
    child_coverage = coverage.Coverage(
        data_file=sys.argv[2], branch=sys.argv[3] == "True", config_file=False,
        source=["scpn_phase_orchestrator.reactor_semantics.producer_registry"],
    )
    child_coverage.start()
from scpn_phase_orchestrator.reactor_semantics.producer_registry import (
    reactor_producer_registry_bytes,
    REACTOR_PRODUCER_REGISTRY_VERSION,
    REACTOR_PRODUCER_REGISTRY_SHA256,
)
try:
    reactor_producer_registry_bytes(
        version=REACTOR_PRODUCER_REGISTRY_VERSION,
        digest=REACTOR_PRODUCER_REGISTRY_SHA256,
    )
except ValueError as error:
    assert str(error) == "packaged producer registry digest mismatch"
else:
    raise AssertionError("corrupt package data accepted")
finally:
    if child_coverage is not None:
        child_coverage.stop()
        child_coverage.save()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(tmp_path),
            str(child_data) if parent_coverage else "",
            str(branch),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if parent_coverage is not None:
        measured = coverage.CoverageData(basename=str(child_data))
        measured.read()
        original = Path(__file__).parents[1] / "src" / package

        def original_source(filename: str) -> str:
            """Map measured copied Python files back to their identical sources."""
            relative = Path(filename).relative_to(tmp_path / package)
            assert Path(filename).read_bytes() == (original / relative).read_bytes()
            return str(original / relative)

        parent_coverage.get_data().update(measured, map_path=original_source)
