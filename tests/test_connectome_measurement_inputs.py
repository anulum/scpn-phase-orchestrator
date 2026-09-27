# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome measurement metadata

"""Exercise connectome source metadata through the public generator."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome


@pytest.mark.parametrize("argument", ["n_regions", "seed"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        "2",
        2.5,
        np.timedelta64(2, "ms"),
        np.datetime64("2026-01-01"),
    ],
)
def test_connectome_rejects_metadata_aliases(argument: str, value: object) -> None:
    """Reject metadata aliases before optional Rust dispatch or cache lookup."""
    with pytest.raises((TypeError, ValueError)):
        load_hcp_connectome(
            cast(int, value) if argument == "n_regions" else 4,
            seed=cast(int, value) if argument == "seed" else 42,
        )


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_connectome_preserves_unsigned_seed_domain(seed: int) -> None:
    """The public generator preserves both endpoints of the seed domain."""
    matrix = load_hcp_connectome(4, seed=seed)
    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))
    np.testing.assert_array_equal(matrix, load_hcp_connectome(4, seed=seed))
