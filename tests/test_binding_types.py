# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.binding.types import (
    BoundaryDef,
    DriverSpec,
    is_valid_channel_id,
    resolve_extractor_type,
)
from scpn_phase_orchestrator.exceptions import ValidationError


def test_driver_spec_channel_config_falls_back_to_empty_and_supports_aliases():
    drivers = DriverSpec(
        physical={"confidence_weight": 0.9},
        informational={"gain": 1.0},
        symbolic={},
        extra={"H": {"window": 32}},
    )

    assert drivers.channel_config("P") == {"confidence_weight": 0.9}
    assert drivers.channel_config("physical") == {"confidence_weight": 0.9}
    assert drivers.channel_config("I") == {"gain": 1.0}
    assert drivers.channel_config("symbolic") == {}
    assert drivers.channel_config("H") == {"window": 32}
    assert drivers.channel_config("Missing") == {}


def test_all_channel_configs_merges_standard_and_extra_channels():
    drivers = DriverSpec(
        physical={"confidence_weight": 0.9},
        informational={"gain": 1.0},
        symbolic={"zeta": 0.4},
        extra={"H": {"window": 32}},
    )

    assert drivers.all_channel_configs()["P"] == {"confidence_weight": 0.9}
    assert drivers.all_channel_configs()["I"] == {"gain": 1.0}
    assert drivers.all_channel_configs()["S"] == {"zeta": 0.4}
    assert drivers.all_channel_configs()["H"] == {"window": 32}


def test_is_valid_channel_id_and_extractor_alias_mapping():
    assert is_valid_channel_id("Risk")
    assert not is_valid_channel_id("bad channel")
    assert resolve_extractor_type("physical") == "hilbert"
    assert resolve_extractor_type("hilbert") == "hilbert"


def test_boundary_lower_upper_inversion_rejected():
    with pytest.raises(
        ValidationError, match=r"lower \(1\.0\) must be < upper \(0\.0\)"
    ):
        BoundaryDef(name="bad", variable="R", lower=1.0, upper=0.0, severity="soft")
