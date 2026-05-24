# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding type omega validation contracts

"""Validation contracts for BindingSpec omega array resolution."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
)


class TestBindingSpecOmegaValidation:
    """Verify that BindingSpec.get_omegas() enforces length consistency
    between declared oscillator_ids and explicit omega lists."""

    def _make_spec(self, oscillator_ids, omegas=None):
        return BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(
                    name="L1",
                    index=0,
                    oscillator_ids=oscillator_ids,
                    omegas=omegas,
                )
            ],
            oscillator_families={},
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[],
        )

    def test_length_mismatch_raises_with_exact_counts(self):
        """Error message must report both the omegas length and oscillator count."""
        spec = self._make_spec(["o1", "o2", "o3"], omegas=[1.0])
        with pytest.raises(ValueError, match="omegas length 1 != oscillator count 3"):
            spec.get_omegas()

    def test_matching_lengths_returns_correct_array(self):
        """Happy path: omegas length matches oscillator count."""
        spec = self._make_spec(["o1", "o2"], omegas=[3.14, 2.71])
        result = spec.get_omegas()
        np.testing.assert_allclose(result, [3.14, 2.71])

    def test_no_omegas_returns_default(self):
        """No omegas specified → returns default per oscillator."""
        spec = self._make_spec(["o1", "o2", "o3"])
        result = spec.get_omegas()
        assert len(result) == 3
