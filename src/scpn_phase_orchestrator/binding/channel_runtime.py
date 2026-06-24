# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — N-channel runtime execution policies

"""Runtime execution of delayed and uncertain N-channel policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from scpn_phase_orchestrator.binding.channel_algebra import (
    ChannelAlgebraReport,
    build_channel_algebra_report,
)
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.upde.metrics import LayerState

__all__ = [
    "ChannelLayerRuntimeEvidence",
    "ChannelRuntimeExecution",
    "ChannelRuntimeExecutor",
]


@dataclass(frozen=True)
class ChannelLayerRuntimeEvidence:
    """Per-layer evidence showing how a channel policy affected execution."""

    layer_index: int
    channel: str
    raw_R: float
    executed_R: float
    raw_psi: float
    executed_psi: float
    delay_policy: str
    uncertainty_policy: str
    evidence_source: str
    confidence_weight: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable layer runtime-evidence record.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the ChannelLayerRuntimeEvidence
            fields.
        """
        return {
            "layer_index": self.layer_index,
            "channel": self.channel,
            "raw_R": self.raw_R,
            "executed_R": self.executed_R,
            "raw_psi": self.raw_psi,
            "executed_psi": self.executed_psi,
            "delay_policy": self.delay_policy,
            "uncertainty_policy": self.uncertainty_policy,
            "evidence_source": self.evidence_source,
            "confidence_weight": self.confidence_weight,
        }


@dataclass(frozen=True)
class ChannelRuntimeExecution:
    """Executed layer states plus audit evidence for one runtime tick."""

    layers: tuple[LayerState, ...]
    evidence: tuple[ChannelLayerRuntimeEvidence, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable runtime execution record.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the ChannelRuntimeExecution
            fields.
        """
        return {
            "layers": [item.to_audit_record() for item in self.evidence],
            "delayed_layers": [
                item.layer_index
                for item in self.evidence
                if item.delay_policy == "hold_last_runtime_evidence"
            ],
            "uncertain_layers": [
                item.layer_index
                for item in self.evidence
                if item.uncertainty_policy == "confidence_weight_runtime_contribution"
            ],
        }


class ChannelRuntimeExecutor:
    """Apply N-channel delay and uncertainty policies to layer diagnostics.

    The executor is intentionally deterministic and non-actuating. It only
    transforms the layer diagnostics consumed by the supervisor and audit log:
    delayed channels contribute the previous tick's layer evidence when
    available, while uncertain channels scale their contribution by an explicit
    driver confidence weight.
    """

    def __init__(
        self,
        *,
        layer_channels: tuple[str, ...],
        report: ChannelAlgebraReport,
        confidence_weights: dict[str, float],
    ) -> None:
        self._layer_channels = layer_channels
        self._report = report
        self._confidence_weights = confidence_weights
        self._last_raw_layers: tuple[LayerState, ...] | None = None

    @classmethod
    def from_spec(cls, spec: BindingSpec) -> ChannelRuntimeExecutor:
        """Build a runtime executor from binding channel metadata.

        Parameters
        ----------
        spec : BindingSpec
            The binding specification supplying channel layer metadata.

        Returns
        -------
        ChannelRuntimeExecutor
            An executor configured with the spec's per-channel runtime policies.
        """
        family_channels = {
            name: family.channel for name, family in spec.oscillator_families.items()
        }
        layer_channels = tuple(
            family_channels.get(layer.family, "P") if layer.family else "P"
            for layer in spec.layers
        )
        return cls(
            layer_channels=layer_channels,
            report=build_channel_algebra_report(spec),
            confidence_weights=_confidence_weights(spec),
        )

    def execute(self, raw_layers: list[LayerState]) -> ChannelRuntimeExecution:
        """Apply delayed/uncertain channel policies to this tick's layer states.

        Parameters
        ----------
        raw_layers : list[LayerState]
            Per-layer states observed for the current tick, one per binding
            layer.

        Returns
        -------
        ChannelRuntimeExecution
            The policy-adjusted layer states with per-layer runtime evidence.

        Raises
        ------
        ValueError
            If ``raw_layers`` does not have one entry per configured binding
            layer.
        """
        if len(raw_layers) != len(self._layer_channels):
            msg = (
                "raw layer count must match binding layer count: "
                f"{len(raw_layers)} != {len(self._layer_channels)}"
            )
            raise ValueError(msg)

        executed: list[LayerState] = []
        evidence: list[ChannelLayerRuntimeEvidence] = []
        delayed = set(self._report.delayed_channels)
        uncertain = set(self._report.uncertain_channels)
        policies = self._report.runtime_policies

        for idx, raw_layer in enumerate(raw_layers):
            channel = self._layer_channels[idx]
            policy = policies.get(channel)
            delay_policy = (
                policy.delay_policy if policy else "use_current_tick_evidence"
            )
            uncertainty_policy = (
                policy.uncertainty_policy
                if policy
                else "deterministic_runtime_contribution"
            )
            if channel in delayed and self._last_raw_layers is not None:
                base_layer = self._last_raw_layers[idx]
                evidence_source = "held_previous_tick"
            elif channel in delayed:
                base_layer = raw_layer
                evidence_source = "current_tick_prime"
            else:
                base_layer = raw_layer
                evidence_source = "current_tick"

            confidence = self._confidence_weights.get(channel, 1.0)
            executed_r = (
                base_layer.R * confidence if channel in uncertain else base_layer.R
            )
            executed_layer = replace(base_layer, R=executed_r)
            executed.append(executed_layer)
            evidence.append(
                ChannelLayerRuntimeEvidence(
                    layer_index=idx,
                    channel=channel,
                    raw_R=raw_layer.R,
                    executed_R=executed_layer.R,
                    raw_psi=raw_layer.psi,
                    executed_psi=executed_layer.psi,
                    delay_policy=delay_policy,
                    uncertainty_policy=uncertainty_policy,
                    evidence_source=evidence_source,
                    confidence_weight=confidence,
                )
            )

        self._last_raw_layers = tuple(raw_layers)
        return ChannelRuntimeExecution(layers=tuple(executed), evidence=tuple(evidence))


def _confidence_weights(spec: BindingSpec) -> dict[str, float]:
    """Return the per-channel confidence weights."""
    weights: dict[str, float] = {}
    for channel, config in spec.drivers.all_channel_configs().items():
        raw = config.get("confidence_weight", config.get("confidence", 1.0))
        weight = raw if isinstance(raw, int | float) else 1.0
        value = float(weight)
        if not isfinite(value):
            value = 1.0
        weights[channel] = min(1.0, max(0.0, value))
    return weights
