# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau coupling tests

"""Tests for amplitude coupling extensions in CouplingBuilder."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder


class TestBuildWithAmplitude:
    def test_knm_r_shape(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        assert cs.knm_r is not None
        assert cs.knm_r.shape == (4, 4)

    def test_knm_r_zero_diagonal(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(8, 0.5, 0.3, 0.3, 0.3)
        assert cs.knm_r is not None
        np.testing.assert_allclose(np.diag(cs.knm_r), 0.0)

    def test_knm_r_nonnegative(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(8, 0.5, 0.3, 0.3, 0.3)
        assert cs.knm_r is not None
        assert np.all(cs.knm_r >= 0.0)

    def test_knm_r_symmetric(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(6, 0.5, 0.3, 0.3, 0.3)
        assert cs.knm_r is not None
        np.testing.assert_allclose(cs.knm_r, cs.knm_r.T)

    def test_knm_r_independent_of_phase(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(4, 0.5, 0.3, 0.8, 0.1)
        assert cs.knm_r is not None
        # knm_r strength should differ from knm
        assert not np.allclose(cs.knm, cs.knm_r)

    def test_phase_coupling_unchanged(self) -> None:
        builder = CouplingBuilder()
        phase_only = builder.build(4, 0.5, 0.3)
        with_amp = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        np.testing.assert_allclose(phase_only.knm, with_amp.knm)


class TestCouplingStateKnmR:
    def test_default_knm_r_none(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build(4, 0.5, 0.3)
        assert cs.knm_r is None

    def test_switch_template_preserves_knm_r(self) -> None:
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        templates = {"custom": np.eye(4)}
        switched = builder.switch_template(cs, "custom", templates)
        assert switched.knm_r is not None
        np.testing.assert_allclose(switched.knm_r, cs.knm_r)
