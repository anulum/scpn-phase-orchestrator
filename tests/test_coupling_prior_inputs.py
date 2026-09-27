# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prior frequency measurement contracts

"""Prior ingress contracts enforced by scpn_phase_orchestrator._array_types."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.prior import UniversalPrior


@pytest.mark.parametrize(
    "dtype", ["U", "S", "timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]"]
)
def test_prior_rejects_text_and_temporal_frequency_arrays(dtype: str) -> None:
    omegas = np.array([1, 2]).astype(dtype)
    with pytest.raises(ValueError, match="omegas"):
        UniversalPrior().estimate_Kc(omegas, 2)


@pytest.mark.parametrize(
    "item",
    [True, "2", b"2", 2 + 0j, np.timedelta64(2, "ms"), np.datetime64("2026-01-01")],
)
def test_prior_rejects_non_numeric_frequency_objects(item: object) -> None:
    omegas = np.array([1, item], dtype=object)
    with pytest.raises(ValueError, match="omegas"):
        UniversalPrior().estimate_Kc(omegas, 2)


def test_prior_preserves_real_numeric_objects_and_critical_coupling() -> None:
    prior = UniversalPrior()
    expected = prior.estimate_Kc(np.array([1.0, 2.0]), 2)
    actual = prior.estimate_Kc(np.array([np.int64(1), np.float64(2)], dtype=object), 2)
    assert actual == expected
    assert actual.K_c_estimate > 0


def test_prior_rejects_real_frequency_objects_outside_float_range() -> None:
    omegas = np.array([10**400, 2], dtype=object)
    with pytest.raises(ValueError, match="finite 1-D frequency vector") as error:
        UniversalPrior().estimate_Kc(omegas, 2)
    assert isinstance(error.value.__cause__, OverflowError)
