# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for optional-dependency import guards

from __future__ import annotations


def test_jax_engine_has_jax_flag():
    """HAS_JAX flag exists regardless of JAX availability."""
    from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

    assert isinstance(HAS_JAX, bool)


def test_nn_module_imports_without_jax():
    """nn/ package imports without JAX (lazy loading)."""
    import scpn_phase_orchestrator.nn as nn_mod

    assert hasattr(nn_mod, "__all__")
