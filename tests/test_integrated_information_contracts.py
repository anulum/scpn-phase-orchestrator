# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Integrated-information contracts

"""Strict integrated-information result and benchmark contract tests."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.information_integration import (
    IntegratedInformationBenchmarkCase,
    IntegratedInformationBenchmarkReport,
    IntegratedInformationResult,
    benchmark_integrated_information_approximations,
    integrated_information,
)


class _UncoercibleArray:
    """Array-like value that rejects NumPy coercion."""

    def __array__(
        self,
        _dtype: object | None = None,
    ) -> np.ndarray[tuple[()], np.dtype[np.object_]]:
        """Raise the same failure shape as a malformed array provider."""
        raise ValueError("array payload unavailable")


def _valid_result() -> IntegratedInformationResult:
    """Return a deterministic valid integrated-information result."""
    phases = np.vstack(
        [
            np.linspace(0.0, 2.0 * np.pi, 96, dtype=np.float64),
            np.linspace(0.1, 2.0 * np.pi + 0.1, 96, dtype=np.float64),
            np.linspace(0.2, 2.0 * np.pi + 0.2, 96, dtype=np.float64),
        ]
    )
    return integrated_information(phases, n_bins=8)


def _valid_report() -> IntegratedInformationBenchmarkReport:
    """Return a deterministic valid approximation benchmark report."""
    return benchmark_integrated_information_approximations(n_samples=128, n_bins=8)


def test_result_rejects_uncoercible_scalar_payload() -> None:
    """Result scalar validation fails closed on uncoercible array providers."""
    with pytest.raises(ValueError, match="finite non-negative real"):
        replace(_valid_result(), phi=cast(float, _UncoercibleArray()))


def test_result_rejects_nonnumeric_pairwise_matrix_payload() -> None:
    """Result validation rejects pairwise matrices that cannot become floats."""
    with pytest.raises(ValueError, match="pairwise_mi must be a numeric matrix"):
        replace(
            _valid_result(),
            pairwise_mi=cast(
                np.ndarray[tuple[int, int], np.dtype[np.float64]],
                np.array([["bad", "payload"], ["payload", "bad"]], dtype=object),
            ),
        )


def test_result_rejects_singleton_pairwise_matrix() -> None:
    """Result validation requires at least two oscillator channels."""
    with pytest.raises(ValueError, match="at least two oscillators"):
        replace(
            _valid_result(),
            pairwise_mi=np.zeros((1, 1), dtype=np.float64),
            minimum_partition=((0,), ()),
        )


def test_benchmark_report_rejects_nonboolean_ordering_flag() -> None:
    """Benchmark reports require a boolean expected-ordering flag."""
    with pytest.raises(ValueError, match="expected_ordering_passed"):
        replace(_valid_report(), expected_ordering_passed=cast(bool, 1))


def test_benchmark_report_rejects_complex_margin() -> None:
    """Benchmark margin validation rejects complex aliases."""
    with pytest.raises(ValueError, match="locked_phi_margin must be real-valued"):
        replace(_valid_report(), locked_phi_margin=cast(float, 0.1 + 0.0j))


def test_benchmark_report_rejects_nonreal_margin() -> None:
    """Benchmark margin validation rejects non-real scalar payloads."""
    with pytest.raises(ValueError, match="modular_total_margin must be a finite real"):
        replace(_valid_report(), modular_total_margin=cast(float, "bad"))


def test_benchmark_report_rejects_non_tuple_cases() -> None:
    """Benchmark reports require a tuple of canonical benchmark cases."""
    with pytest.raises(ValueError, match="cases must contain benchmark cases"):
        replace(
            _valid_report(),
            cases=cast(tuple[IntegratedInformationBenchmarkCase, ...], []),
        )


def test_benchmark_report_rejects_non_case_entry() -> None:
    """Benchmark reports reject objects that are not benchmark cases."""
    report = _valid_report()
    cases = cast(
        tuple[IntegratedInformationBenchmarkCase, ...],
        (object(), *report.cases[1:]),
    )

    with pytest.raises(ValueError, match="cases must contain benchmark cases"):
        replace(report, cases=cases)


def test_benchmark_report_rejects_empty_case_identity_fields() -> None:
    """Benchmark cases require non-empty names and descriptions."""
    report = _valid_report()
    empty_name = replace(report.cases[0], name="")
    empty_description = replace(report.cases[0], description="")

    with pytest.raises(ValueError, match="case names"):
        replace(report, cases=(empty_name, *report.cases[1:]))
    with pytest.raises(ValueError, match="case descriptions"):
        replace(report, cases=(empty_description, *report.cases[1:]))


def test_normalise_phi_handles_non_positive_log_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normalisation helper guards a non-positive log scale."""
    monkeypatch.setattr(
        "scpn_phase_orchestrator.monitor.information_integration.np.log",
        lambda _value: 0.0,
    )

    from scpn_phase_orchestrator.monitor.information_integration import _normalise_phi

    assert _normalise_phi(1.0, 2) == 0.0
