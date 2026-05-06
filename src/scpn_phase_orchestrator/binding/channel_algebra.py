# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding channel algebra summaries

"""Deterministic N-channel algebra summaries for binding specs."""

from __future__ import annotations

from dataclasses import dataclass

from scpn_phase_orchestrator.binding.types import BindingSpec

__all__ = [
    "ChannelAlgebraReport",
    "ChannelCouplingEdge",
    "build_channel_algebra_report",
]


@dataclass(frozen=True)
class ChannelCouplingEdge:
    """JSON-safe cross-channel coupling edge."""

    source: str
    target: str
    strength: float
    mode: str
    template: str | None

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable coupling-edge record."""
        return {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
            "mode": self.mode,
            "template": self.template,
        }


@dataclass(frozen=True)
class ChannelAlgebraReport:
    """Deterministic channel algebra view for audit, replay, and reporting."""

    channels: tuple[str, ...]
    declared_channels: tuple[str, ...]
    required_channels: tuple[str, ...]
    optional_channels: tuple[str, ...]
    derived_channels: tuple[str, ...]
    delayed_channels: tuple[str, ...]
    uncertain_channels: tuple[str, ...]
    runtime_evidence_channels: tuple[str, ...]
    missing_required_channels: tuple[str, ...]
    supervisor_visible_channels: tuple[str, ...]
    coupling_participating_channels: tuple[str, ...]
    replay_semantics: dict[str, str]
    channel_groups: dict[str, tuple[str, ...]]
    channel_membership: dict[str, tuple[str, ...]]
    coupling_edges: tuple[ChannelCouplingEdge, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable channel algebra record."""
        return {
            "channels": list(self.channels),
            "declared_channels": list(self.declared_channels),
            "required_channels": list(self.required_channels),
            "optional_channels": list(self.optional_channels),
            "derived_channels": list(self.derived_channels),
            "delayed_channels": list(self.delayed_channels),
            "uncertain_channels": list(self.uncertain_channels),
            "runtime_evidence_channels": list(self.runtime_evidence_channels),
            "missing_required_channels": list(self.missing_required_channels),
            "supervisor_visible_channels": list(self.supervisor_visible_channels),
            "coupling_participating_channels": list(
                self.coupling_participating_channels
            ),
            "replay_semantics": dict(sorted(self.replay_semantics.items())),
            "channel_groups": {
                name: list(channels)
                for name, channels in sorted(self.channel_groups.items())
            },
            "channel_membership": {
                channel: list(groups)
                for channel, groups in sorted(self.channel_membership.items())
            },
            "coupling_edges": [edge.to_audit_record() for edge in self.coupling_edges],
        }


def build_channel_algebra_report(spec: BindingSpec) -> ChannelAlgebraReport:
    """Build a deterministic N-channel algebra report from a binding spec.

    The report is a read-only structural view. It does not validate or mutate
    the binding; callers should still run `validate_binding_spec()` for gates.
    """
    channels = tuple(sorted(spec.used_channels()))
    declared_channels = tuple(sorted(spec.channels))
    runtime_evidence_channels = _runtime_evidence_channels(spec)
    required_channels = tuple(
        sorted(
            channel
            for channel, channel_spec in spec.channels.items()
            if channel_spec.required
        )
    )
    optional_channels = tuple(
        sorted(
            channel
            for channel, channel_spec in spec.channels.items()
            if not channel_spec.required
        )
    )
    derived_channels = tuple(
        sorted(
            channel
            for channel, channel_spec in spec.channels.items()
            if channel_spec.derived_from
            or channel_spec.replay_semantics == "derived"
            or channel_spec.derive_rule is not None
        )
    )
    delayed_channels = tuple(
        sorted(
            channel
            for channel, channel_spec in spec.channels.items()
            if _mentions_policy_marker(
                (
                    channel_spec.role,
                    channel_spec.metric_semantics,
                    channel_spec.replay_semantics,
                ),
                ("delayed", "delay", "lagged", "external"),
            )
        )
    )
    uncertain_channels = tuple(
        sorted(
            channel
            for channel, channel_spec in spec.channels.items()
            if _mentions_policy_marker(
                (
                    channel_spec.role,
                    channel_spec.metric_semantics,
                    channel_spec.replay_semantics,
                ),
                ("uncertain", "uncertainty", "probabilistic", "confidence"),
            )
        )
    )
    missing_required_channels = tuple(
        sorted(
            channel
            for channel in required_channels
            if channel not in runtime_evidence_channels
            and channel not in derived_channels
        )
    )
    supervisor_visible_channels = tuple(
        sorted(
            channel
            for channel in channels
            if spec.channels.get(channel) is None
            or spec.channels[channel].supervisor_visibility
        )
    )
    coupling_participating_channels = tuple(
        sorted(
            channel
            for channel in channels
            if spec.channels.get(channel) is None
            or spec.channels[channel].coupling_participation
        )
    )
    channel_groups = {
        name: tuple(group.channels)
        for name, group in sorted(spec.channel_groups.items())
    }
    channel_membership = _channel_membership(channels, channel_groups)
    replay_semantics = {
        channel: spec.channels[channel].replay_semantics
        for channel in declared_channels
    }
    coupling_edges = tuple(
        ChannelCouplingEdge(
            source=coupling.source,
            target=coupling.target,
            strength=coupling.strength,
            mode=coupling.mode,
            template=coupling.template,
        )
        for coupling in spec.cross_channel_couplings
    )
    return ChannelAlgebraReport(
        channels=channels,
        declared_channels=declared_channels,
        required_channels=required_channels,
        optional_channels=optional_channels,
        derived_channels=derived_channels,
        delayed_channels=delayed_channels,
        uncertain_channels=uncertain_channels,
        runtime_evidence_channels=runtime_evidence_channels,
        missing_required_channels=missing_required_channels,
        supervisor_visible_channels=supervisor_visible_channels,
        coupling_participating_channels=coupling_participating_channels,
        replay_semantics=replay_semantics,
        channel_groups=channel_groups,
        channel_membership=channel_membership,
        coupling_edges=coupling_edges,
    )


def _runtime_evidence_channels(spec: BindingSpec) -> tuple[str, ...]:
    family_channels = {family.channel for family in spec.oscillator_families.values()}
    configured_driver_channels = {
        channel
        for channel, config in spec.drivers.all_channel_configs().items()
        if bool(config)
    }
    return tuple(sorted(family_channels | configured_driver_channels))


def _mentions_policy_marker(
    values: tuple[str | None, ...],
    markers: tuple[str, ...],
) -> bool:
    text = " ".join(value.lower() for value in values if value)
    return any(marker in text for marker in markers)


def _channel_membership(
    channels: tuple[str, ...],
    channel_groups: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    memberships: dict[str, list[str]] = {channel: [] for channel in channels}
    for group_name, group_channels in channel_groups.items():
        for channel in group_channels:
            memberships.setdefault(channel, []).append(group_name)
    return {
        channel: tuple(sorted(groups))
        for channel, groups in sorted(memberships.items())
    }
