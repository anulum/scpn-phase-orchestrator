# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — dVOC plant builder refuses non-finite parameters

"""``nan <= 0`` is false: a NaN frequency, ratio or step built a NaN plant."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.dvoc_oscillation_damping import (
    underdamped_oscillator,
)

_GOOD = {"frequency_hz": 1.0, "damping_ratio": 0.05, "dt": 0.02}


@pytest.mark.parametrize("name", sorted(_GOOD))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_non_finite_or_non_real_parameter_is_refused(name: str, bad: object) -> None:
    kwargs = dict(_GOOD)
    kwargs[name] = bad
    with pytest.raises(ValueError, match=name):
        underdamped_oscillator(**kwargs)  # type: ignore[arg-type]


def test_numpy_scalars_build_the_same_plant() -> None:
    plain = underdamped_oscillator(**_GOOD)
    numpy_args = underdamped_oscillator(
        frequency_hz=np.float64(1.0),
        damping_ratio=np.float32(0.05),
        dt=np.float64(0.02),
    )
    np.testing.assert_allclose(numpy_args[0], plain[0], rtol=1e-6)
    np.testing.assert_allclose(numpy_args[1], plain[1], rtol=1e-6)
    assert np.all(np.isfinite(plain[0]))
