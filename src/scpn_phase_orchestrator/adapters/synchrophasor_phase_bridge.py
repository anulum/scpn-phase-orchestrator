# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 synchrophasor phase bridge

"""Map decoded IEEE C37.118.2 PMU phasor measurements to oscillator phase states.

A phasor measurement unit already reports a phase-resolved quantity: each voltage
or current phasor carries a magnitude and an angle, and the frame carries the
measured line frequency. Unlike the scalar SCADA tags of the OPC-UA bridge — a
raw waveform from which a phase must be *extracted* — a PMU phasor's angle *is*
the instantaneous phase, so this bridge reads it directly rather than running a
Hilbert or zero-crossing extractor:

* ``theta`` is the phasor angle, canonicalised to ``[0, 2*pi)``. For a
  rectangular phasor it is ``atan2(imag, real)`` (scale-independent, so the
  PHUNIT conversion factor never enters the angle); for a floating-point polar
  phasor it is the reported angle in radians.
* ``omega`` is the instantaneous angular frequency, ``2*pi`` times the frame's
  absolute measured frequency in hertz.
* ``amplitude`` is the phasor magnitude in engineering units — integer
  components are scaled by the PHUNIT ``10**-5`` V/A-per-bit factor; float
  components are already in engineering units.
* ``quality`` is derived only from the STAT word's verified data-error field
  (bits 15-14) and time-sync bit (bit 13).

Integer *polar* phasors are an honest boundary: the standard scales an integer
polar angle differently from the magnitude, and the open-source references
disagree on that scaling, so rather than guess an angle unit this bridge raises
for integer polar phasors instead of emitting a fabricated angle. The bridge is
review-only: it produces phase states for observation and never actuates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_non_negative_int,
)
from scpn_phase_orchestrator.oscillators.base import PhaseState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
        ConfigurationFrame2,
        DataFrame,
        PmuConfiguration,
        PmuMeasurement,
    )

__all__ = [
    "C37118PhaseBridge",
    "PhasorBinding",
]

#: STAT data-error field (bits 15-14); non-zero means the measurement is invalid.
_STAT_DATA_ERROR = 0xC000
#: STAT time-synchronisation bit (bit 13); set means the PMU is out of sync.
_STAT_SYNC = 0x2000
#: Quality for a valid, time-synchronised measurement.
_QUALITY_GOOD = 1.0
#: Quality for a valid but out-of-sync measurement.
_QUALITY_UNSYNCED = 0.5
#: Quality for a measurement flagged as a data error.
_QUALITY_INVALID = 0.0
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PhasorBinding:
    """Bind one PMU phasor channel to an SPO oscillator.

    Attributes
    ----------
    oscillator : str
        Oscillator name the phasor's phase state is bound to.
    pmu_index : int
        Index of the PMU within the configuration/data frame (default ``0``).
    phasor_index : int
        Index of the phasor channel within the PMU block (default ``0``).
    """

    oscillator: str
    pmu_index: int = 0
    phasor_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "oscillator",
            require_non_empty_str(self.oscillator, field="oscillator"),
        )
        require_non_negative_int(self.pmu_index, field="pmu_index")
        require_non_negative_int(self.phasor_index, field="phasor_index")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the binding.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the binding fields.
        """
        return {
            "oscillator": self.oscillator,
            "pmu_index": self.pmu_index,
            "phasor_index": self.phasor_index,
        }


def _stat_quality(stat: int) -> float:
    """Return a phase-state quality in ``[0, 1]`` from a STAT flag word."""
    if stat & _STAT_DATA_ERROR:
        return _QUALITY_INVALID
    if stat & _STAT_SYNC:
        return _QUALITY_UNSYNCED
    return _QUALITY_GOOD


def _phasor_theta_amplitude(
    pmu: PmuConfiguration,
    measurement: PmuMeasurement,
    phasor_index: int,
) -> tuple[float, float]:
    """Return ``(theta, amplitude)`` for one phasor in engineering units.

    Parameters
    ----------
    pmu : PmuConfiguration
        The PMU's decoded configuration (format flags and PHUNIT factors).
    measurement : PmuMeasurement
        The decoded measurement block for that PMU.
    phasor_index : int
        Index of the phasor channel within the block.

    Returns
    -------
    tuple[float, float]
        The phase angle in ``[0, 2*pi)`` and the magnitude in engineering units.

    Raises
    ------
    ValueError
        If ``phasor_index`` is out of range, or the phasor is integer polar
        (whose angle scaling this bridge does not model).
    """
    if not 0 <= phasor_index < len(measurement.phasors):
        raise ValueError(
            f"phasor_index {phasor_index} out of range for "
            f"{len(measurement.phasors)} phasors"
        )
    first, second = measurement.phasors[phasor_index]
    if pmu.phasor_polar:
        if not pmu.phasor_float:
            raise ValueError(
                "integer polar phasor angle scaling is not modelled; "
                "use a rectangular or floating-point PMU stream"
            )
        magnitude, angle = first, second
    else:
        angle = math.atan2(second, first)
        magnitude = math.hypot(first, second)
        if not pmu.phasor_float:
            scale = _phasor_scale(pmu, phasor_index)
            magnitude *= scale
    return angle % _TWO_PI, magnitude


def _phasor_scale(pmu: PmuConfiguration, phasor_index: int) -> float:
    """Return the PHUNIT engineering scale for an integer phasor component."""
    if phasor_index < len(pmu.phasor_units):
        return pmu.phasor_units[phasor_index].volts_or_amperes_per_bit
    return 1.0


@dataclass
class C37118PhaseBridge:
    """Review-only bridge from decoded PMU phasors to oscillator phase states.

    Attributes
    ----------
    bindings : tuple[PhasorBinding, ...]
        The phasor-to-oscillator bindings; oscillator names must be unique.
    non_actuating : bool
        Always ``True`` — the bridge observes and never drives hardware.
    execution_disabled : bool
        Always ``True`` — no control action is emitted from this bridge.
    """

    bindings: tuple[PhasorBinding, ...]
    non_actuating: bool = field(default=True, init=False)
    execution_disabled: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.bindings:
            raise ValueError("at least one binding must be declared")
        names = [binding.oscillator for binding in self.bindings]
        if len(set(names)) != len(names):
            raise ValueError("binding oscillator names must be unique")

    @classmethod
    def from_bindings(cls, bindings: Sequence[PhasorBinding]) -> C37118PhaseBridge:
        """Build a bridge from a sequence of phasor bindings.

        Parameters
        ----------
        bindings : Sequence[PhasorBinding]
            The phasor-to-oscillator bindings.

        Returns
        -------
        C37118PhaseBridge
            A configured, review-only bridge.
        """
        return cls(bindings=tuple(bindings))

    def extract_phases(
        self,
        config: ConfigurationFrame2,
        frames: Sequence[DataFrame],
    ) -> dict[str, PhaseState]:
        """Map the most recent frame's phasors to per-oscillator phase states.

        Parameters
        ----------
        config : ConfigurationFrame2
            The configuration describing each PMU's measurement layout.
        frames : Sequence[DataFrame]
            The decoded DATA frames; the last frame provides the current state.

        Returns
        -------
        dict[str, PhaseState]
            The latest instantaneous phase state per bound oscillator.

        Raises
        ------
        ValueError
            If ``frames`` is empty, a binding's PMU index is out of range, or a
            bound phasor cannot be interpreted (integer polar angle).
        """
        if not frames:
            raise ValueError("at least one DATA frame is required")
        latest = frames[-1]
        phases: dict[str, PhaseState] = {}
        for binding in self.bindings:
            if not 0 <= binding.pmu_index < len(config.pmus):
                raise ValueError(
                    f"pmu_index {binding.pmu_index} out of range for "
                    f"{len(config.pmus)} PMUs"
                )
            if binding.pmu_index >= len(latest.measurements):
                raise ValueError(
                    f"pmu_index {binding.pmu_index} out of range for a frame "
                    f"with {len(latest.measurements)} measurements"
                )
            pmu = config.pmus[binding.pmu_index]
            measurement = latest.measurements[binding.pmu_index]
            theta, amplitude = _phasor_theta_amplitude(
                pmu, measurement, binding.phasor_index
            )
            phases[binding.oscillator] = PhaseState(
                theta=theta,
                omega=_TWO_PI * measurement.frequency_hz,
                amplitude=amplitude,
                quality=_stat_quality(measurement.stat),
                channel="P",
                node_id=binding.oscillator,
            )
        return phases

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the bridge.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the bridge configuration and
            its review-only posture.
        """
        return {
            "bindings": [binding.to_audit_record() for binding in self.bindings],
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
        }
