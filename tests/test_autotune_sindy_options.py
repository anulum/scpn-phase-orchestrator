# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase-SINDy operator options tests

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.autotune.discovery import TimeSeriesDiscoveryConfig
from scpn_phase_orchestrator.autotune.sindy_confidence import (
    DEFAULT_SINDY_CONFIDENCE_POLICY,
    SindyConfidencePolicy,
)
from scpn_phase_orchestrator.autotune.sindy_options import (
    DEFAULT_SINDY_OPTIONS,
    SindyOptions,
)


def test_defaults_match_the_conservative_shared_configuration() -> None:
    options = SindyOptions()

    assert options.phase_sindy_threshold == 0.05
    assert options.confidence_policy is DEFAULT_SINDY_CONFIDENCE_POLICY
    assert DEFAULT_SINDY_OPTIONS.phase_sindy_threshold == 0.05


def test_to_discovery_config_carries_only_the_phase_sindy_threshold() -> None:
    options = SindyOptions(phase_sindy_threshold=0.2)

    config = options.to_discovery_config()

    assert isinstance(config, TimeSeriesDiscoveryConfig)
    assert config.phase_sindy_threshold == 0.2
    # The other discovery thresholds keep their defaults.
    assert config.sindy_threshold == TimeSeriesDiscoveryConfig().sindy_threshold
    assert (
        config.correlation_threshold
        == TimeSeriesDiscoveryConfig().correlation_threshold
    )


def test_custom_confidence_policy_is_carried_verbatim() -> None:
    policy = SindyConfidencePolicy(min_r_squared=0.7, min_samples_per_parameter=8.0)

    options = SindyOptions(confidence_policy=policy)

    assert options.confidence_policy is policy


@pytest.mark.parametrize("bad", [-0.1, -1.0, math.inf, math.nan, -math.inf])
def test_negative_or_non_finite_threshold_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="phase_sindy_threshold"):
        SindyOptions(phase_sindy_threshold=bad)


def test_zero_threshold_is_accepted() -> None:
    options = SindyOptions(phase_sindy_threshold=0.0)

    assert options.phase_sindy_threshold == 0.0


def test_options_are_frozen() -> None:
    options = SindyOptions()

    with pytest.raises(AttributeError):
        options.phase_sindy_threshold = 0.1  # type: ignore[misc]
