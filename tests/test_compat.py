# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import math

from scpn_phase_orchestrator._compat import HAS_RUST, TWO_PI


def test_two_pi_value():
    assert abs(TWO_PI - 2.0 * math.pi) < 1e-15


def test_has_rust_is_bool():
    assert isinstance(HAS_RUST, bool)
