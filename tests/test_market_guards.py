# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Market monitor validation and fallback guards

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
)


class TestSeriesGuards:
    def test_extract_phase_rejects_non_coercible_series(self) -> None:
        series = np.array([["a"], ["b"]], dtype=object)
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            extract_phase(series)

    def test_extract_phase_rejects_zero_channel_series(self) -> None:
        with pytest.raises(ValueError, match="at least one channel"):
            extract_phase(np.zeros((3, 0)))


class TestPhaseMatrixGuards:
    def test_order_parameter_rejects_non_coercible_phases(self) -> None:
        phases = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match=r"finite \(T, N\) array"):
            market_order_parameter(phases)


class TestSignalVectorGuards:
    def test_detect_regimes_rejects_non_coercible_signal(self) -> None:
        with pytest.raises(ValueError, match="finite one-dimensional array"):
            detect_regimes(np.array(["a", "b"], dtype=object))

    def test_detect_regimes_rejects_two_dimensional_signal(self) -> None:
        with pytest.raises(ValueError, match="must be one-dimensional"):
            detect_regimes(np.zeros((2, 2)))

    def test_detect_regimes_rejects_non_real_threshold(self) -> None:
        with pytest.raises(ValueError, match="sync_threshold must be a finite real"):
            detect_regimes(np.array([0.5, 0.6]), sync_threshold="high")


class TestRegimeNumpyFallback:
    def test_python_fallback_classifies_regimes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        # Replace the compiled kernel with an empty stub so the inline
        # ``from spo_kernel import detect_regimes_rust`` raises ImportError and
        # the NumPy masking fallback runs. Deleting the attribute directly is
        # unreliable on read-only C-extension modules across CI interpreters.
        monkeypatch.setitem(sys.modules, "spo_kernel", types.ModuleType("spo_kernel"))

        regimes = detect_regimes(
            np.array([0.1, 0.5, 0.9], dtype=np.float64),
            sync_threshold=0.7,
            desync_threshold=0.3,
        )

        np.testing.assert_array_equal(regimes, np.array([0, 1, 2], dtype=np.int32))
