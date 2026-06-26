# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for environment readiness diagnostics

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime import cli, doctor
from scpn_phase_orchestrator.runtime.doctor import (
    REQUIRED_PYTHON,
    DependencyCheck,
    DoctorReport,
    render_report,
    run_environment_diagnostics,
)


class TestDependencyCheck:
    def test_status_ok_when_available(self) -> None:
        check = DependencyCheck("numpy", "core", True, True, "numpy 2.0", "2.0")
        assert check.status == "ok"

    def test_status_missing_when_required_absent(self) -> None:
        check = DependencyCheck("numpy", "core", True, False, "absent")
        assert check.status == "missing"

    def test_status_warn_when_optional_absent(self) -> None:
        check = DependencyCheck("jax", "nn", False, False, "absent")
        assert check.status == "warn"

    def test_to_record_has_deterministic_keys(self) -> None:
        record = DependencyCheck(
            "rust", "backend", False, True, "ready", "0.5"
        ).to_record()
        assert list(record) == [
            "name",
            "category",
            "required",
            "available",
            "status",
            "version",
            "detail",
        ]
        assert record["status"] == "ok"
        assert record["version"] == "0.5"


def _check(name: str, *, required: bool, available: bool) -> DependencyCheck:
    return DependencyCheck(name, "core", required, available, "detail")


class TestDoctorReport:
    def test_ok_report_status_and_exit_code(self) -> None:
        report = DoctorReport(
            checks=(_check("python", required=True, available=True),),
            python_version="3.12.3",
            platform="Linux x86_64",
        )
        assert report.ok is True
        assert report.status == "pass"
        assert report.exit_code == 0
        assert report.missing_required == ()

    def test_missing_required_fails(self) -> None:
        report = DoctorReport(
            checks=(
                _check("numpy", required=True, available=False),
                _check("jax", required=False, available=False),
            ),
            python_version="3.12.3",
        )
        assert report.ok is False
        assert report.status == "fail"
        assert report.exit_code == 1
        assert [c.name for c in report.missing_required] == ["numpy"]
        assert [c.name for c in report.missing_optional] == ["jax"]

    def test_audit_record_counts(self) -> None:
        report = DoctorReport(
            checks=(
                _check("numpy", required=True, available=True),
                _check("scipy", required=True, available=False),
                _check("jax", required=False, available=True),
                _check("optax", required=False, available=False),
            ),
            python_version="3.12.3",
            platform="Linux x86_64",
        )
        record = report.to_audit_record()
        assert record["status"] == "fail"
        assert record["required_present"] == 1
        assert record["required_total"] == 2
        assert record["optional_present"] == 1
        assert record["optional_total"] == 2
        assert record["missing_required"] == ["scipy"]
        assert record["missing_optional"] == ["optax"]
        assert record["version"] == "1.0.0"
        # JSON-serialisable and stable.
        assert json.loads(json.dumps(record)) == record


class TestModulePresent:
    def test_present_module(self) -> None:
        assert doctor._module_present("numpy") is True

    def test_absent_module(self) -> None:
        assert doctor._module_present("scpn_no_such_module_xyz") is False

    def test_missing_parent_does_not_raise(self) -> None:
        # Dotted name whose parent is absent: find_spec raises ModuleNotFoundError,
        # which must be swallowed into a False rather than propagating.
        assert doctor._module_present("scpn_no_such_module_xyz.child") is False


class TestDistributionVersion:
    def test_present(self) -> None:
        assert doctor._distribution_version("numpy") is not None

    def test_absent_returns_none(self) -> None:
        assert doctor._distribution_version("scpn-no-such-dist-xyz") is None


class TestRustProbe:
    def test_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "_module_present", lambda name: True)
        monkeypatch.setattr(doctor, "_distribution_version", lambda name: "0.5.10")
        check = doctor._check_rust()
        assert check.available is True
        assert "0.5.10" in check.detail

    def test_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "_module_present", lambda name: False)
        check = doctor._check_rust()
        assert check.available is False
        assert "not importable" in check.detail


class TestFindRepoRoot:
    def test_detects_checkout(self, tmp_path: Path) -> None:
        (tmp_path / "go").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        assert doctor._find_repo_root(nested / "mod.py") == tmp_path

    def test_no_checkout_returns_none(self, tmp_path: Path) -> None:
        assert doctor._find_repo_root(tmp_path / "deep" / "mod.py") is None


class TestPythonCheck:
    def test_in_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            doctor.sys, "version_info", SimpleNamespace(major=3, minor=12, micro=3)
        )
        check = doctor._check_python()
        assert check.available is True
        assert check.required is True
        assert "satisfies" in check.detail

    def test_below_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            doctor.sys, "version_info", SimpleNamespace(major=3, minor=9, micro=18)
        )
        check = doctor._check_python()
        assert check.available is False
        assert "outside the supported window" in check.detail

    def test_at_upper_bound_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            doctor.sys, "version_info", SimpleNamespace(major=3, minor=14, micro=0)
        )
        check = doctor._check_python()
        assert check.available is False

    def test_required_window_matches_pyproject(self) -> None:
        assert REQUIRED_PYTHON == ((3, 10), (3, 14))


class TestGoProbe:
    def test_prebuilt_libraries(self, tmp_path: Path) -> None:
        go_dir = tmp_path / "go"
        go_dir.mkdir()
        (go_dir / "libhodge.so").write_bytes(b"")
        (go_dir / "libnpe.so").write_bytes(b"")
        check = doctor._check_go(tmp_path)
        assert check.available is True
        assert "libhodge.so" in check.detail
        assert "libnpe.so" in check.detail

    def test_toolchain_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/go")
        check = doctor._check_go(tmp_path)  # no ./go dir
        assert check.available is True
        assert "toolchain" in check.detail

    def test_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
        check = doctor._check_go(tmp_path)
        assert check.available is False
        assert "no 'go' toolchain" in check.detail

    def test_repo_root_none(self) -> None:
        assert doctor._go_shared_libraries(None) == []


class TestMojoProbe:
    def test_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda name: "/opt/mojo")
        check = doctor._check_mojo()
        assert check.available is True
        assert "/opt/mojo" in check.detail

    def test_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
        check = doctor._check_mojo()
        assert check.available is False


class TestAdapterSurfaceProbe:
    def test_fmi_and_hybrid_surfaces_are_reported(self) -> None:
        report = run_environment_diagnostics()
        adapters = {c.name: c for c in report.checks if c.category == "adapter"}
        assert adapters["fmi-cosimulation"].available is True
        assert "CoSimulationSlave" in adapters["fmi-cosimulation"].detail
        assert adapters["hybrid-cocompiler"].available is True
        assert (
            "build_hybrid_cocompiler_manifest" in adapters["hybrid-cocompiler"].detail
        )

    def test_adapter_surface_missing_export_is_warning(self) -> None:
        check = doctor._check_adapter_surface(
            name="broken-adapter",
            module_name="json",
            symbols=("missing_export",),
            detail="broken adapter",
        )
        assert check.available is False
        assert check.required is False
        assert check.category == "adapter"
        assert check.status == "warn"
        assert "missing_export" in check.detail


class TestRunDiagnostics:
    def test_includes_python_and_core(self) -> None:
        report = run_environment_diagnostics()
        names = {c.name for c in report.checks}
        assert "python" in names
        assert {"numpy", "scipy", "pyyaml", "click", "protobuf", "urllib3"} <= names
        # Backends and a representative optional extra are probed.
        assert {"rust", "julia", "go", "mojo"} <= names
        assert "jax" in names

    def test_core_present_in_test_env(self) -> None:
        # The test environment installs the core dependencies, so the report
        # must be ready and exit zero.
        report = run_environment_diagnostics()
        assert report.ok is True
        assert report.exit_code == 0

    def test_missing_required_dependency_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = doctor._module_present

        def fake(import_name: str) -> bool:
            return False if import_name == "numpy" else real(import_name)

        monkeypatch.setattr(doctor, "_module_present", fake)
        report = run_environment_diagnostics()
        assert report.ok is False
        assert "numpy" in [c.name for c in report.missing_required]

    def test_explicit_repo_root(self, tmp_path: Path) -> None:
        report = run_environment_diagnostics(repo_root=tmp_path)
        go_check = next(c for c in report.checks if c.name == "go")
        assert go_check.category == "backend"


class TestRenderReport:
    def test_pass_render(self) -> None:
        report = DoctorReport(
            checks=(
                _check("python", required=True, available=True),
                _check("jax", required=False, available=False),
            ),
            python_version="3.12.3",
            platform="Linux x86_64",
        )
        lines = render_report(report)
        text = "\n".join(lines)
        assert "PASS" in text
        assert "python" in text
        assert "[warn]" in text
        assert "jax" in text

    def test_fail_render(self) -> None:
        report = DoctorReport(
            checks=(_check("numpy", required=True, available=False),),
            python_version="3.12.3",
        )
        text = "\n".join(render_report(report))
        assert "FAIL" in text
        assert "[MISS]" in text
        assert "numpy" in text


class TestCli:
    def test_doctor_text(self) -> None:
        result = CliRunner().invoke(cli.main, ["doctor"])
        assert result.exit_code == 0
        assert "environment diagnostics" in result.output
        assert "python" in result.output

    def test_doctor_json(self) -> None:
        result = CliRunner().invoke(cli.main, ["doctor", "--json-out"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["report"] == "environment-diagnostics"
        assert payload["status"] == "pass"
        assert "checks" in payload

    def test_doctor_failure_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        failing = DoctorReport(
            checks=(_check("numpy", required=True, available=False),),
            python_version="3.12.3",
        )
        monkeypatch.setattr(
            cli.diagnostics, "run_environment_diagnostics", lambda: failing
        )
        result = CliRunner().invoke(cli.main, ["doctor"])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_doctor_failure_json_exit_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failing = DoctorReport(
            checks=(_check("numpy", required=True, available=False),),
            python_version="3.12.3",
        )
        monkeypatch.setattr(
            cli.diagnostics, "run_environment_diagnostics", lambda: failing
        )
        result = CliRunner().invoke(cli.main, ["doctor", "--json-out"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["status"] == "fail"
