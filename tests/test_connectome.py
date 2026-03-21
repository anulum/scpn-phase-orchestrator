# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — HCP connectome loader tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome


def test_output_shape():
    knm = load_hcp_connectome(20)
    assert knm.shape == (20, 20)


def test_symmetric():
    knm = load_hcp_connectome(40)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)


def test_zero_diagonal():
    knm = load_hcp_connectome(30)
    np.testing.assert_allclose(np.diag(knm), 0.0, atol=1e-15)


def test_non_negative():
    knm = load_hcp_connectome(50)
    assert np.all(knm >= 0.0)


def test_intra_stronger_than_inter():
    """Intra-hemispheric coupling should be stronger than inter on average."""
    n = 40
    knm = load_hcp_connectome(n)
    half = n // 2
    intra_mean = (knm[:half, :half].sum() + knm[half:, half:].sum()) / (2 * half * (half - 1))
    inter_mean = knm[:half, half:].sum() / (half * (n - half))
    assert intra_mean > inter_mean


def test_deterministic():
    """Same n_regions → same matrix (seeded RNG)."""
    a = load_hcp_connectome(24)
    b = load_hcp_connectome(24)
    np.testing.assert_array_equal(a, b)


def test_small_n_raises():
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        load_hcp_connectome(1)
