# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE numerics general-Real path tests

"""Tests for the general-Real validation path of ``check_stability``.

``check_stability`` has a fast path for exact ``float``/``int`` arguments and a
general path that runs the ``_validate_*`` helpers for other ``Real`` types such
as numpy scalars. These tests drive that general path with ``numpy.float64``
inputs: a non-zero bound, a zero bound, and a negative bound that fails closed.
"""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.numerics as _numerics
from scpn_phase_orchestrator.upde.numerics import check_stability

assert _numerics is not None


def test_general_real_path_returns_bound_for_non_zero_derivative() -> None:
    result = check_stability(np.float64(0.01), np.float64(1.0), np.float64(0.5))
    # dt * (1.0 + 0.5) = 0.015 < pi -> stable.
    assert result is True


def test_general_real_path_is_unstable_for_a_large_step() -> None:
    result = check_stability(np.float64(10.0), np.float64(1.0), np.float64(0.5))
    # dt * 1.5 = 15.0 >= pi -> not stable.
    assert result is False


def test_general_real_path_returns_true_for_zero_derivative() -> None:
    result = check_stability(np.float64(0.5), np.float64(0.0), np.float64(0.0))
    assert result is True


def test_general_real_path_rejects_a_negative_frequency_bound() -> None:
    with pytest.raises(
        ValueError, match="max_omega must be a finite non-negative real"
    ):
        check_stability(np.float64(0.01), np.float64(-1.0), np.float64(0.5))


def test_general_real_path_rejects_a_non_positive_timestep() -> None:
    with pytest.raises(ValueError, match="dt must be a finite positive real"):
        check_stability(np.float64(0.0), np.float64(1.0), np.float64(0.5))
