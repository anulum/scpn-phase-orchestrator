# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE lazy export tests

"""Behavioural tests for the lazy public exports in ``upde.__init__``."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde as upde


def test_lazy_export_resolves_order_parameter_pipeline_entrypoint() -> None:
    phases = np.array([0.0, 0.0])

    r, psi = upde.compute_order_parameter(phases)

    assert r == pytest.approx(1.0, abs=1e-12)
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_unknown_lazy_export_raises_clear_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'not_a_upde_export'"):
        upde.__getattr__("not_a_upde_export")


def test_dir_returns_declared_public_exports() -> None:
    exported = upde.__dir__()

    assert exported is upde.__all__
    assert "UPDEEngine" in exported
    assert "compute_order_parameter" in exported
