# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation lazy export tests

"""Behavioural tests for the lazy public exports in ``actuation.__init__``."""

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.actuation as actuation
from scpn_phase_orchestrator.actuation import constraints, mapper


def test_lazy_exports_resolve_to_their_defining_modules() -> None:
    """Each public name loads the class from the module that defines it."""
    assert actuation.ActionProjector is constraints.ActionProjector
    assert actuation.ActuationMapper is mapper.ActuationMapper
    assert actuation.ControlAction is mapper.ControlAction


def test_unknown_actuation_export_raises_a_clear_attribute_error() -> None:
    """A name outside the export table is refused with the module named."""
    with pytest.raises(AttributeError, match="has no attribute 'not_an_actuator'"):
        actuation.__getattr__("not_an_actuator")


def test_dir_lists_every_lazy_export_before_it_is_loaded() -> None:
    """Introspection shows the public names even when none is loaded yet."""
    listing = dir(actuation)
    assert listing == sorted(listing)
    assert set(actuation.__all__) <= set(listing)
