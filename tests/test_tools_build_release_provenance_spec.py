# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for build_release_provenance_spec.py

"""Unit tests for ``tools/build_release_provenance_spec.py``.

Covers the deterministic digest computation for subjects, byproducts, and resolved
dependencies; the derivation of the build identity, invocation, and builder version
from the CI environment; the empty-artefact and missing-environment guards; and the
``main`` entry point writing a spec that ``spo provenance-attest`` accepts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "build_release_provenance_spec.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_build_release_provenance_spec_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "GITHUB_SERVER_URL": "https://github.com/",
        "GITHUB_REPOSITORY": "anulum/scpn-phase-orchestrator",
        "GITHUB_REF": "refs/tags/v0.9.0",
        "GITHUB_SHA": "d" * 40,
        "GITHUB_RUN_ID": "1234567890",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_WORKFLOW": "Release",
        "RUNNER_OS": "Linux",
        "PROVENANCE_PYTHON_VERSION": "3.12.3",
    }
    env.update(overrides)
    return env


def _artifact(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestSha256File:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        path = _artifact(tmp_path, "a.bin", b"payload-bytes")
        assert mod._sha256_file(path) == hashlib.sha256(b"payload-bytes").hexdigest()

    def test_hashes_large_multichunk_file(self, tmp_path: Path) -> None:
        data = b"x" * (mod._CHUNK * 2 + 7)
        path = _artifact(tmp_path, "big.bin", data)
        assert mod._sha256_file(path) == hashlib.sha256(data).hexdigest()


class TestRequireEnv:
    def test_returns_value(self) -> None:
        assert mod._require_env({"K": "v"}, "K") == "v"

    def test_raises_on_missing(self) -> None:
        with pytest.raises(KeyError, match="GITHUB_SHA is missing"):
            mod._require_env({}, "GITHUB_SHA")

    def test_raises_on_empty(self) -> None:
        with pytest.raises(KeyError, match="GITHUB_SHA is missing"):
            mod._require_env({"GITHUB_SHA": ""}, "GITHUB_SHA")


class TestBuildSpec:
    def test_subjects_are_digest_pinned(self, tmp_path: Path) -> None:
        art = _artifact(tmp_path, "pkg-0.9.0.tar.gz", b"sdist")
        spec = mod.build_release_provenance_spec(
            artifacts=[art], sbom_files=[], lock_files=[], env=_env()
        )
        assert spec["subjects"] == [
            {"name": "pkg-0.9.0.tar.gz", "sha256": hashlib.sha256(b"sdist").hexdigest()}
        ]

    def test_byproducts_and_resolved_dependencies(self, tmp_path: Path) -> None:
        art = _artifact(tmp_path, "pkg.tar.gz", b"sdist")
        sbom = _artifact(tmp_path, "sbom.json", b"{}")
        lock = _artifact(tmp_path, "pqc-lock.txt", b"cryptography==49.0.0")
        spec = mod.build_release_provenance_spec(
            artifacts=[art], sbom_files=[sbom], lock_files=[lock], env=_env()
        )
        byproduct = spec["byproducts"][0]
        assert byproduct["name"] == "sbom.json"
        assert byproduct["sha256"] == hashlib.sha256(b"{}").hexdigest()
        assert byproduct["uri"].endswith("/releases/download/v0.9.0/sbom.json")
        dep = spec["resolved_dependencies"][0]
        assert dep["name"] == "pqc-lock.txt"
        assert dep["uri"] == (
            "https://github.com/anulum/scpn-phase-orchestrator/blob/"
            + "d" * 40
            + "/requirements/pqc-lock.txt"
        )

    def test_build_identity_and_version(self, tmp_path: Path) -> None:
        art = _artifact(tmp_path, "pkg.tar.gz", b"sdist")
        spec = mod.build_release_provenance_spec(
            artifacts=[art], sbom_files=[], lock_files=[], env=_env()
        )
        assert spec["build_type"] == mod.BUILD_TYPE
        assert spec["external_parameters"] == {
            "ref": "refs/tags/v0.9.0",
            "repository": "anulum/scpn-phase-orchestrator",
            "workflow": "Release",
            "source_commit": "d" * 40,
        }
        assert spec["builder_id"].endswith(
            ".github/workflows/release.yml@refs/tags/v0.9.0"
        )
        assert spec["invocation_id"].endswith("/actions/runs/1234567890/attempts/2")
        assert spec["builder_version"] == {
            "github_run_id": "1234567890",
            "github_run_attempt": "2",
            "runner_os": "Linux",
            "python": "3.12.3",
        }

    def test_defaults_run_attempt_and_workflow(self, tmp_path: Path) -> None:
        art = _artifact(tmp_path, "pkg.tar.gz", b"sdist")
        env = _env()
        del env["GITHUB_RUN_ATTEMPT"]
        del env["GITHUB_WORKFLOW"]
        spec = mod.build_release_provenance_spec(
            artifacts=[art], sbom_files=[], lock_files=[], env=env
        )
        assert spec["builder_version"]["github_run_attempt"] == "1"
        assert spec["external_parameters"]["workflow"] == "Release"

    def test_rejects_empty_artifacts(self) -> None:
        with pytest.raises(ValueError, match="at least one release artefact"):
            mod.build_release_provenance_spec(
                artifacts=[], sbom_files=[], lock_files=[], env=_env()
            )

    def test_missing_required_env_raises(self, tmp_path: Path) -> None:
        art = _artifact(tmp_path, "pkg.tar.gz", b"sdist")
        env = _env()
        del env["GITHUB_SHA"]
        with pytest.raises(KeyError, match="GITHUB_SHA"):
            mod.build_release_provenance_spec(
                artifacts=[art], sbom_files=[], lock_files=[], env=env
            )


class TestMain:
    def test_writes_spec_consumable_by_attest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = _artifact(tmp_path, "pkg-0.9.0.tar.gz", b"sdist")
        sbom = _artifact(tmp_path, "sbom.json", b"{}")
        lock = _artifact(tmp_path, "pqc-lock.txt", b"cryptography==49.0.0")
        for key, value in _env().items():
            monkeypatch.setenv(key, value)
        output = tmp_path / "spec.json"
        rc = mod.main(
            [
                "--artifact",
                str(art),
                "--sbom",
                str(sbom),
                "--lock",
                str(lock),
                "--output",
                str(output),
            ]
        )
        assert rc == 0
        spec = json.loads(output.read_text(encoding="utf-8"))
        # The written spec must carry every field spo provenance-attest reads.
        for key in (
            "subjects",
            "build_type",
            "external_parameters",
            "resolved_dependencies",
            "byproducts",
            "builder_id",
            "invocation_id",
            "builder_version",
        ):
            assert key in spec
