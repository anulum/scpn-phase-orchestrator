# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy measurement ingress

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor._te_go import (
    phase_te_go,
    te_matrix_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_julia import (
    phase_te_julia,
    te_matrix_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_mojo import (
    phase_te_mojo,
    te_matrix_mojo,
)
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)

FloatArray: TypeAlias = NDArray[np.float64]
PairFn: TypeAlias = Callable[[FloatArray, FloatArray, int], float]
MatrixFn: TypeAlias = Callable[[FloatArray, int, int, int], FloatArray]


@pytest.mark.parametrize(
    "pair", [phase_transfer_entropy, phase_te_go, phase_te_julia, phase_te_mojo]
)
@pytest.mark.parametrize("argument", ["source", "target"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]"]
)
def test_pairwise_ingress_rejects_temporal_phase_units(
    pair: PairFn, argument: str, dtype: str
) -> None:
    real = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    invalid = real.astype(dtype)
    with pytest.raises(ValueError, match=argument):
        pair(
            invalid if argument == "source" else real,
            invalid if argument == "target" else real,
            2,
        )


@pytest.mark.parametrize(
    "pair", [phase_transfer_entropy, phase_te_go, phase_te_julia, phase_te_mojo]
)
@pytest.mark.parametrize("argument", ["source", "target"])
def test_pairwise_ingress_rejects_temporal_numeric_objects(
    pair: PairFn, argument: str
) -> None:
    real = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    invalid = real.astype(object)
    invalid[1] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match=argument):
        pair(
            invalid if argument == "source" else real,
            invalid if argument == "target" else real,
            2,
        )


@pytest.mark.parametrize("matrix", [te_matrix_go, te_matrix_julia, te_matrix_mojo])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]"]
)
def test_direct_matrix_ingress_rejects_temporal_flat_payloads(
    matrix: MatrixFn, dtype: str
) -> None:
    invalid = np.array([0, 1, 2, 1, 2, 3]).astype(dtype)
    with pytest.raises(ValueError, match="phase_series"):
        matrix(invalid, 2, 3, 2)


def test_public_matrix_rejects_temporal_object_phases() -> None:
    series = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=object)
    series[0, 1] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match="phase_series"):
        transfer_entropy_matrix(series, n_bins=2)


def test_public_pair_preserves_real_numeric_object_entropy() -> None:
    source = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    target = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    expected = phase_transfer_entropy(source, target, n_bins=2)
    actual = phase_transfer_entropy(
        source.astype(object), target.astype(object), n_bins=2
    )
    assert actual == pytest.approx(expected, abs=1e-12)
    assert 0 <= actual <= np.log(2)
