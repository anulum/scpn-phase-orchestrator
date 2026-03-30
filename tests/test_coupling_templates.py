# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling template tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet


def _tpl(name, n=3, fill=0.0, desc=""):
    knm = np.full((n, n), fill)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    return KnmTemplate(name=name, knm=knm, alpha=alpha, description=desc)


# ---------------------------------------------------------------------------
# KnmTemplate: frozen data integrity
# ---------------------------------------------------------------------------


class TestKnmTemplateIntegrity:
    """Verify that templates preserve coupling matrices exactly
    and cannot be mutated after creation."""

    def test_arrays_preserved_exactly(self):
        knm = np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]])
        alpha = np.array([[0.0, 0.05, -0.1], [-0.05, 0.0, 0.0], [0.1, 0.0, 0.0]])
        tpl = KnmTemplate("precise", knm=knm, alpha=alpha, description="test")
        np.testing.assert_array_equal(tpl.knm, knm)
        np.testing.assert_array_equal(tpl.alpha, alpha)

    def test_frozen_rejects_mutation(self):
        tpl = _tpl("frozen")
        with pytest.raises(AttributeError):
            tpl.name = "changed"
        with pytest.raises(AttributeError):
            tpl.description = "changed"

    def test_metadata_preserved(self):
        tpl = KnmTemplate("cortical_ring", np.eye(2), np.zeros((2, 2)),
                           "Ring topology from Paper 12")
        assert tpl.name == "cortical_ring"
        assert tpl.description == "Ring topology from Paper 12"


# ---------------------------------------------------------------------------
# KnmTemplateSet: registry CRUD operations
# ---------------------------------------------------------------------------


class TestKnmTemplateSetRegistry:
    """Verify the template registry: add, retrieve, list, overwrite, error
    reporting with available names."""

    def test_add_and_retrieve_roundtrip(self):
        ts = KnmTemplateSet()
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        tpl = KnmTemplate("sym", knm=knm, alpha=np.zeros((2, 2)), description="symmetric")
        ts.add(tpl)
        retrieved = ts.get("sym")
        np.testing.assert_array_equal(retrieved.knm, knm)
        assert retrieved.description == "symmetric"

    def test_list_names_ordering(self):
        ts = KnmTemplateSet()
        ts.add(_tpl("beta"))
        ts.add(_tpl("alpha"))
        ts.add(_tpl("gamma"))
        names = ts.list_names()
        assert set(names) == {"alpha", "beta", "gamma"}
        assert len(names) == 3

    def test_overwrite_replaces_completely(self):
        """Re-adding with same name must replace knm, alpha, and description."""
        ts = KnmTemplateSet()
        ts.add(KnmTemplate("x", np.eye(2), np.zeros((2, 2)), "first"))
        new_knm = np.ones((2, 2))
        new_alpha = np.full((2, 2), 0.1)
        ts.add(KnmTemplate("x", new_knm, new_alpha, "second"))
        result = ts.get("x")
        assert result.description == "second"
        np.testing.assert_array_equal(result.knm, new_knm)
        np.testing.assert_array_equal(result.alpha, new_alpha)

    def test_multiple_templates_independent(self):
        """Templates with different names must be independent."""
        ts = KnmTemplateSet()
        ts.add(_tpl("ring", n=4, fill=0.3))
        ts.add(_tpl("star", n=4, fill=0.8))
        ring = ts.get("ring")
        star = ts.get("star")
        assert not np.allclose(ring.knm, star.knm), (
            "Different templates must have different coupling matrices"
        )

    def test_empty_set_operations(self):
        ts = KnmTemplateSet()
        assert ts.list_names() == []
        with pytest.raises(KeyError):
            ts.get("anything")


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


class TestTemplateErrorReporting:
    """Verify that error messages help users identify what went wrong
    and what options are available."""

    def test_missing_template_lists_available(self):
        ts = KnmTemplateSet()
        ts.add(_tpl("alpha"))
        ts.add(_tpl("beta"))
        with pytest.raises(KeyError, match="alpha") as exc_info:
            ts.get("gamma")
        # Error message should mention both available templates
        assert "beta" in str(exc_info.value)
        assert "gamma" in str(exc_info.value)

    def test_missing_from_empty_shows_none(self):
        ts = KnmTemplateSet()
        with pytest.raises(KeyError, match="none"):
            ts.get("anything")

    def test_error_includes_requested_name(self):
        ts = KnmTemplateSet()
        ts.add(_tpl("x"))
        with pytest.raises(KeyError, match="nonexistent"):
            ts.get("nonexistent")


# ---------------------------------------------------------------------------
# Physics contract: templates used in simulation
# ---------------------------------------------------------------------------


class TestTemplatePhysicsIntegration:
    """Verify that templates can be used to switch coupling matrices
    at runtime — the primary use case for regime-dependent coupling."""

    def test_switching_templates_changes_dynamics(self):
        """Switching from weak to strong coupling template must change
        the UPDE step output."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        ts = KnmTemplateSet()
        ts.add(_tpl("weak", n=n, fill=0.1))
        ts.add(_tpl("strong", n=n, fill=2.0))

        eng = UPDEEngine(n, dt=0.01)
        phases = np.linspace(0, np.pi, n)
        omegas = np.ones(n)

        weak_tpl = ts.get("weak")
        strong_tpl = ts.get("strong")

        result_weak = eng.step(phases, omegas, weak_tpl.knm, 0.0, 0.0, weak_tpl.alpha)
        result_strong = eng.step(phases, omegas, strong_tpl.knm, 0.0, 0.0, strong_tpl.alpha)

        # Strong coupling must produce larger phase corrections
        diff_weak = np.max(np.abs(result_weak - phases))
        diff_strong = np.max(np.abs(result_strong - phases))
        assert diff_strong > diff_weak, (
            f"Strong coupling should produce larger phase changes: "
            f"weak={diff_weak:.4f}, strong={diff_strong:.4f}"
        )

    def test_template_alpha_affects_phase_lag(self):
        """Non-zero alpha in a template should shift phase dynamics
        compared to zero alpha."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 3
        knm = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
        alpha_zero = np.zeros((n, n))
        alpha_lag = np.array([[0.0, 0.5, -0.5], [-0.5, 0.0, 0.3], [0.5, -0.3, 0.0]])

        eng = UPDEEngine(n, dt=0.01)
        phases = np.array([0.0, np.pi / 3, np.pi])
        omegas = np.zeros(n)

        result_no_lag = eng.step(phases, omegas, knm, 0.0, 0.0, alpha_zero)
        result_with_lag = eng.step(phases, omegas, knm, 0.0, 0.0, alpha_lag)

        assert not np.allclose(result_no_lag, result_with_lag), (
            "Phase lag alpha must change the dynamics"
        )
