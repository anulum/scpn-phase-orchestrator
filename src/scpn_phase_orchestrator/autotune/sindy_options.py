# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase-SINDy operator options

"""Operator-facing options for the phase-SINDy discovery honesty surface.

A single bundle carries the two knobs an operator turns when running phase-SINDy
discovery through a binding proposal or the CLI: the sparsity threshold that
decides which coupling coefficients survive, and the confidence policy that
decides how strong a fit must be before its recovered structure is called
``discovered``. Keeping them together means the binding proposal and the CLI
configure discovery the same way without duplicating the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from scpn_phase_orchestrator.autotune.discovery import TimeSeriesDiscoveryConfig
from scpn_phase_orchestrator.autotune.sindy_confidence import (
    DEFAULT_SINDY_CONFIDENCE_POLICY,
    SindyConfidencePolicy,
)


@dataclass(frozen=True)
class SindyOptions:
    """Operator configuration for phase-SINDy discovery and its confidence.

    Parameters
    ----------
    phase_sindy_threshold : float
        Sparsity threshold below which a coupling coefficient is dropped from
        the phase-SINDy fit. Must be finite and non-negative; defaults to the
        discovery default of ``0.05``.
    confidence_policy : SindyConfidencePolicy
        Thresholds separating a credible discovery from weak evidence. Defaults
        to the conservative shared policy.
    """

    phase_sindy_threshold: float = 0.05
    confidence_policy: SindyConfidencePolicy = DEFAULT_SINDY_CONFIDENCE_POLICY

    def __post_init__(self) -> None:
        """Validate the threshold is finite and non-negative."""
        threshold = float(self.phase_sindy_threshold)
        if not isfinite(threshold) or threshold < 0.0:
            raise ValueError("phase_sindy_threshold must be finite and non-negative")
        object.__setattr__(self, "phase_sindy_threshold", threshold)

    def to_discovery_config(self) -> TimeSeriesDiscoveryConfig:
        """Return the discovery config carrying the phase-SINDy threshold.

        Only the phase-SINDy threshold is overridden; the other discovery
        thresholds keep their defaults.

        Returns
        -------
        TimeSeriesDiscoveryConfig
            A config with ``phase_sindy_threshold`` set from these options.
        """
        return TimeSeriesDiscoveryConfig(
            phase_sindy_threshold=self.phase_sindy_threshold
        )


DEFAULT_SINDY_OPTIONS = SindyOptions()
