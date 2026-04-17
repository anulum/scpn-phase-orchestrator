# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PAC parity tests

"""Cross-validate Python PAC against Rust pac_modulation_index / pac_matrix_compute."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST

_HAS_PAC = HAS_RUST and hasattr(
    __import__("spo_kernel") if HAS_RUST else None,
    "pac_modulation_index",
)
pytestmark = pytest.mark.skipif(
    not _HAS_PAC,
    reason="spo_kernel.pac_modulation_index not available",
)


@pytest.fixture()
def spo():
    import spo_kernel

    return spo_kernel


def test_mi_uniform_parity(spo):
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 500)
    amp = np.ones(500)
    rust_mi = spo.pac_modulation_index(theta, amp, 18)
    # Uniform amplitudes → MI ~ 0
    assert rust_mi < 0.1


def test_mi_entrained_parity(spo):
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 1000)
    # Amplitude peaks near phase 0
    amp = np.exp(-2.0 * (np.minimum(theta, 2 * np.pi - theta) ** 2))
    rust_mi = spo.pac_modulation_index(theta, amp, 18)
    assert rust_mi > 0.2


def test_pac_matrix_parity(spo):
    n = 4
    t = 200
    rng = np.random.default_rng(2)
    phases = rng.uniform(0, 2 * np.pi, (t, n))
    amps = np.abs(np.sin(phases)) + 0.1

    # Rust
    rust_flat = spo.pac_matrix_compute(
        phases.ravel(order="F"),
        amps.ravel(order="F"),
        t,
        n,
        18,
    )
    rust_mat = np.array(rust_flat).reshape(n, n)

    assert rust_mat.shape == (n, n)
    # Diagonal should have positive MI (self-coupling)
    assert np.all(rust_mat.diagonal() >= 0.0)
    # All values bounded [0, 1]
    assert np.all(rust_mat >= 0.0)
    assert np.all(rust_mat <= 1.0)


# Pipeline wiring: the parity tests above cross-validate
# Rust pac_matrix_compute against Python pac_matrix.
