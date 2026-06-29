# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL controller synthesis branch tests

"""Branch tests for STL controller-candidate synthesis.

Covers the unparseable-spec guard (an automaton whose spec is not the builtin
simple STL grammar) and the equality-predicate ``restore`` control direction.
"""

from __future__ import annotations

import dataclasses

import pytest

import scpn_phase_orchestrator.monitor.stl.controller as _controller
from scpn_phase_orchestrator.monitor.stl import (
    synthesise_stl_controller_candidates,
    synthesise_stl_monitoring_automaton,
)

assert _controller is not None


def test_synthesis_rejects_an_unparseable_spec() -> None:
    trace = {"R": [0.8, 0.2, 0.5]}
    automaton = synthesise_stl_monitoring_automaton("always (R >= 0.3)", trace)
    # A spec outside the builtin simple grammar fails to parse, so synthesis
    # cannot proceed.
    unparseable = dataclasses.replace(automaton, spec="bounded_until(R, K, 3)")

    with pytest.raises(ValueError, match="builtin simple STL syntax only"):
        synthesise_stl_controller_candidates(unparseable, trace)


def test_equality_predicate_yields_a_restore_direction() -> None:
    # R never equals 0.5, so the == predicate is violated and a candidate is
    # synthesised with the "restore" control direction.
    trace = {"R": [0.1, 0.2, 0.3]}
    automaton = synthesise_stl_monitoring_automaton("always (R == 0.5)", trace)

    synthesis = synthesise_stl_controller_candidates(automaton, trace)

    assert synthesis.candidates
    assert any(candidate.direction == "restore" for candidate in synthesis.candidates)
