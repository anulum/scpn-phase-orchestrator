# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling lazy export tests

"""Behavioural tests for lazy public exports in ``coupling.__init__``."""

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.coupling as coupling


def test_lazy_coupling_builder_export_builds_pipeline_matrix() -> None:
    builder = coupling.CouplingBuilder()

    state = builder.build(n_layers=4, base_strength=0.6, decay_alpha=0.2)
    knm = state.knm

    assert knm.shape == (4, 4)
    assert knm.diagonal().sum() == pytest.approx(0.0, abs=1e-12)
    assert state.alpha.shape == (4, 4)
    assert state.active_template == "default"
    coupling.validate_knm(knm)


def test_unknown_coupling_export_raises_clear_attribute_error() -> None:
    with pytest.raises(
        AttributeError,
        match="has no attribute 'not_a_coupling_export'",
    ):
        coupling.__getattr__("not_a_coupling_export")


def test_dir_returns_declared_coupling_public_exports() -> None:
    exported = coupling.__dir__()

    assert exported is coupling.__all__
    assert "CouplingBuilder" in exported
    assert "validate_knm" in exported
