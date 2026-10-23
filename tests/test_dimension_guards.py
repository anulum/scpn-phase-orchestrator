# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension result and output guards

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    _validate_ci_values,
    _validate_ky_dimension,
)
from tests.test_dimension_julia_absence import run_source_profile


def _result(slope: object) -> CorrelationDimensionResult:
    return CorrelationDimensionResult(
        D2=1.0,
        epsilons=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        C_eps=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        slope=cast("NDArray[np.float64]", slope),
        scaling_range=(0.1, 0.3),
    )


class TestCorrelationDimensionResultGuards:
    def test_rejects_boolean_slope(self) -> None:
        with pytest.raises(ValueError, match="slope must not contain boolean values"):
            _result(np.array([True, False]))

    def test_rejects_non_coercible_slope(self) -> None:
        with pytest.raises(ValueError, match="slope must be a finite one-dimensional"):
            _result(np.array(["a", "b"], dtype=object))


class TestOutputContracts:
    """Direct validation of untrusted compute-backend output."""

    def test_correlation_integral_rejects_non_coercible_output(self) -> None:
        bad = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_ci_values(bad, expected_size=2)

    def test_kaplan_yorke_rejects_complex_output(self) -> None:
        with pytest.raises(ValueError, match="must be real-valued"):
            _validate_ky_dimension(complex(1.0, 1.0), n_exponents=3)


class TestKaplanYorkeBackendFallback:
    @pytest.mark.native_runtime
    def test_falls_back_to_exact_reference_when_backend_raises(
        self, tmp_path: Path
    ) -> None:
        """Loss of execution permission after actual Mojo admission preserves KY."""
        run_source_profile(tmp_path, "unusable-mojo")


class TestCorrelationBackendFallback:
    @pytest.mark.native_runtime
    def test_correlation_dimension_uses_numpy_when_backend_raises(
        self, tmp_path: Path
    ) -> None:
        """The real admitted binary becoming unusable preserves the public D2 grid."""
        run_source_profile(tmp_path, "unusable-mojo")
