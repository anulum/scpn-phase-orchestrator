# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL monitor contract tests

"""Focused public contracts for the runtime STL monitor module."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Sequence
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Protocol, cast

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor.stl.monitor as monitor_module


class _LoadedMonitorModule(Protocol):
    """Protocol for an isolated loaded copy of the STL monitor module."""

    HAS_RTAMT: bool
    STLMonitor: type[monitor_module.STLMonitor]


class _FakeStlDiscreteTimeSpecification:
    """Minimal rtamt-compatible discrete-time specification for monitor tests."""

    result: ClassVar[object] = [[0.0, 1.25], [1.0, -0.5], [2.0, 0.75]]

    def __init__(self) -> None:
        self.spec = ""
        self.declared: list[tuple[str, str]] = []
        self.parsed = False

    def declare_var(self, name: str, var_type: str) -> None:
        """Record a declared signal variable."""
        self.declared.append((name, var_type))

    def parse(self) -> None:
        """Record that parsing was requested."""
        self.parsed = True

    def evaluate(self, datasets: dict[str, list[float]]) -> object:
        """Return the configured robustness result for ``datasets``."""
        assert datasets["time"] == [0.0, 1.0, 2.0]
        assert datasets["x"] == [0.0, 1.0, 2.0]
        assert datasets["y"] == [1.0, 0.5, 0.0]
        return self.result


class _FakeRtamtModule:
    """Minimal module-shaped object exposing rtamt's monitor constructor."""

    StlDiscreteTimeSpecification = _FakeStlDiscreteTimeSpecification


class TestOptionalRtamtBackend:
    """Contracts for the optional rtamt execution path."""

    def test_monitor_uses_rtamt_backend_for_unsupported_builtin_syntax(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The monitor delegates non-builtin formulas to the rtamt backend."""
        _FakeStlDiscreteTimeSpecification.result = [
            [0.0, 1.25],
            [1.0, -0.5],
            [2.0, 0.75],
        ]
        monkeypatch.setattr(monitor_module, "rtamt", _FakeRtamtModule())
        monitor = monitor_module.STLMonitor("x until y")

        robustness = monitor.evaluate({"x": [0.0, 1.0, 2.0], "y": [1.0, 0.5, 0.0]})
        repeated = monitor.evaluate({"x": [0.0, 1.0, 2.0], "y": [1.0, 0.5, 0.0]})
        result = monitor.evaluate_result({"x": [0.0, 1.0, 2.0], "y": [1.0, 0.5, 0.0]})

        assert robustness == pytest.approx(-0.5)
        assert repeated == pytest.approx(-0.5)
        assert result.backend == "rtamt"
        assert result.satisfied is False

    def test_monitor_accepts_scalar_rtamt_robustness(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The monitor accepts scalar robustness values from rtamt."""
        _FakeStlDiscreteTimeSpecification.result = 0.75
        monkeypatch.setattr(monitor_module, "rtamt", _FakeRtamtModule())
        monitor = monitor_module.STLMonitor("x until y")

        assert monitor.evaluate({"x": [0.0, 1.0, 2.0], "y": [1.0, 0.5, 0.0]}) == 0.75

    def test_monitor_import_falls_back_when_rtamt_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The module records optional-backend absence during import."""
        original_import = builtins.__import__

        def guarded_import(
            name: str,
            globals_: object = None,
            locals_: object = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> object:
            if name == "rtamt":
                raise ImportError("rtamt unavailable for test")
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        module_name = "_stl_monitor_import_probe"
        module_path = Path(monitor_module.__file__ or "")
        loader = SourceFileLoader(module_name, str(module_path))
        probe = ModuleType(module_name)
        probe.__file__ = str(module_path)
        sys.modules[module_name] = probe
        try:
            loader.exec_module(probe)
        finally:
            sys.modules.pop(module_name, None)
        unavailable = cast(_LoadedMonitorModule, probe)

        assert unavailable.HAS_RTAMT is False
        monitor = unavailable.STLMonitor("x until y")
        with pytest.raises(ImportError, match="rtamt is required"):
            monitor.evaluate({"x": [1.0], "y": [0.0]})


class TestTraceValidationContracts:
    """Public trace-validation contracts for builtin STL evaluation."""

    def test_monitor_rejects_missing_predicate_signal(self) -> None:
        """Builtin evaluation rejects traces missing a predicate signal."""
        monitor = monitor_module.STLMonitor("always (missing >= 0.0)")

        with pytest.raises(ValueError, match="trace missing signal 'missing'"):
            monitor.evaluate({"R": [0.5]})

    def test_monitor_rejects_non_1d_signal_payload(self) -> None:
        """Builtin evaluation rejects non-vector signal payloads."""
        monitor = monitor_module.STLMonitor("always (R >= 0.0)")

        with pytest.raises(ValueError, match="1-D numeric signal"):
            monitor.evaluate({"R": cast(list[float], [[0.5], [0.6]])})

    def test_monitor_rejects_non_numeric_signal_payload(self) -> None:
        """Builtin evaluation rejects non-numeric signal samples."""
        monitor = monitor_module.STLMonitor("always (R >= 0.0)")

        with pytest.raises(ValueError, match="must be numeric"):
            monitor.evaluate({"R": cast(list[float], ["bad"])})

    def test_monitor_rejects_numpy_boolean_signal_payload(self) -> None:
        """Builtin evaluation rejects NumPy boolean arrays."""
        monitor = monitor_module.STLMonitor("always (R >= 0.0)")

        with pytest.raises(ValueError, match="boolean"):
            monitor.evaluate({"R": cast(list[float], np.asarray([True, False]))})

    def test_monitor_rejects_numpy_complex_signal_payload(self) -> None:
        """Builtin evaluation rejects NumPy complex arrays."""
        monitor = monitor_module.STLMonitor("always (R >= 0.0)")

        with pytest.raises(ValueError, match="real-valued"):
            monitor.evaluate({"R": cast(list[float], np.asarray([1.0 + 0.0j]))})

    def test_equal_predicate_robustness_is_zero_at_exact_match(self) -> None:
        """Equality predicates report zero robustness only at exact matches."""
        monitor = monitor_module.STLMonitor("always (R == 0.5)")

        assert monitor.evaluate({"R": [0.5, 0.5]}) == 0.0
        assert monitor.evaluate({"R": [0.5, 0.25]}) < 0.0


class TestMonitorDefensiveHelpers:
    """Defensive helper branches not reachable through the public parser."""

    def test_evaluate_simple_rejects_unknown_temporal_operator(self) -> None:
        """The simple evaluator rejects unsupported parsed temporal operators."""
        parsed = ("once", [("R", ">=", 0.3)])

        with pytest.raises(ValueError, match="unsupported STL temporal operator"):
            monitor_module._evaluate_simple(parsed, {"R": [0.5]})

    def test_predicate_robustness_rejects_unknown_comparison_operator(self) -> None:
        """Predicate robustness rejects unsupported comparison operators."""
        with pytest.raises(ValueError, match="unsupported STL comparison operator"):
            monitor_module._predicate_robustness("R", "!=", 0.3, {"R": [0.5]})

    def test_threshold_formatting_is_stable_for_integer_and_decimal_values(
        self,
    ) -> None:
        """Threshold formatting keeps guard strings deterministic."""
        assert monitor_module._format_threshold(10.0) == "10"
        assert monitor_module._format_threshold(0.125) == "0.125"

    def test_non_empty_validation_rejects_blank_projection_fields(self) -> None:
        """The shared STL non-empty validator rejects blank projection fields."""
        with pytest.raises(ValueError, match="projection action"):
            monitor_module._require_non_empty("   ", "projection action")

    def test_boolean_alias_detection_covers_typed_bool_and_float_arrays(self) -> None:
        """The boolean-alias helper handles typed bool and plain float dtypes."""
        assert monitor_module._contains_boolean_alias(np.array([True, False])) is True
        assert monitor_module._contains_boolean_alias(np.array([1.0, 2.0])) is False

    def test_complex_alias_detection_covers_typed_complex_and_float_arrays(
        self,
    ) -> None:
        """The complex-alias helper handles typed complex and plain float dtypes."""
        assert monitor_module._contains_complex_alias(np.array([1.0 + 2.0j])) is True
        assert monitor_module._contains_complex_alias(np.array([1.0, 2.0])) is False
