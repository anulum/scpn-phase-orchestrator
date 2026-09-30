# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding channel admission contracts

"""Channel declaration refusals and recovery through typed and file bindings."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.binding.channel_algebra import build_channel_algebra_report
from scpn_phase_orchestrator.binding.channel_runtime import ChannelRuntimeExecutor
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
    ChannelGroupSpec,
    ChannelSpec,
    CrossChannelCouplingSpec,
    OscillatorFamily,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.upde.metrics import LayerState


def _binding() -> BindingSpec:
    """Load the repository's minimal domain with an optional risk channel."""
    root = Path(__file__).resolve().parents[1]
    return replace(
        load_binding_spec(root / "domainpacks/minimal_domain/binding_spec.yaml"),
        channels={"Risk": ChannelSpec(role="risk", required=False)},
    )


def _write_binding(spec: BindingSpec, path: Path) -> BindingSpec:
    """Serialise typed channel declarations through the public JSON loader."""
    payload = asdict(spec)
    payload["drivers"] = {
        "physical": spec.drivers.physical,
        "informational": spec.drivers.informational,
        "symbolic": spec.drivers.symbolic,
        **(spec.drivers.extra or {}),
    }
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return load_binding_spec(path)


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        pytest.param(
            ChannelSpec(role="risk", required=False, replay_semantics="log"),
            "replay_semantics must be one of",
            id="unsupported-replay",
        ),
        pytest.param(
            ChannelSpec(
                role="risk",
                replay_semantics="derived",
                derived_from=["Risk"],
                derive_rule="lag(Risk)",
            ),
            "derived_from must not include itself",
            id="self-source",
        ),
        pytest.param(
            ChannelSpec(
                role="risk",
                replay_semantics="derived",
                derived_from=["Unknown"],
                derive_rule="phase(Unknown)",
            ),
            "derived_from references unknown channel 'Unknown'",
            id="unknown-source",
        ),
        pytest.param(
            ChannelSpec(role="risk", replay_semantics="derived", derived_from=["P"]),
            "derive_rule is required when derived_from is set",
            id="missing-rule",
        ),
        pytest.param(
            ChannelSpec(
                role="risk",
                derived_from=["P"],
                derive_rule="phase(P)",
            ),
            "derived_from channels must use replay_semantics='derived'",
            id="wrong-source-replay",
        ),
        pytest.param(
            ChannelSpec(role="risk", required=False, replay_semantics="derived"),
            "replay_semantics='derived' requires derived_from",
            id="missing-sources",
        ),
        pytest.param(
            ChannelSpec(role="risk", required=False, derive_rule="phase(P)"),
            "derive_rule requires derived_from",
            id="orphan-rule",
        ),
        pytest.param(
            ChannelSpec(role="risk"),
            "required channel must be backed by an oscillator family or driver",
            id="missing-evidence",
        ),
    ],
)
def test_channel_declaration_refusal_preserves_binding_and_recovers(
    channel: ChannelSpec,
    expected: str,
    tmp_path: Path,
) -> None:
    """Report each cross-field fault identically before and after JSON loading."""
    valid = _binding()
    bad = replace(valid, channels={"Risk": channel})
    snapshot = asdict(bad)
    path = tmp_path / "binding.json"
    loaded = _write_binding(bad, path)
    loaded_snapshot = asdict(loaded)
    source = path.read_bytes()
    expected_errors = [f"channel 'Risk': {expected}"]
    for candidate in (bad, loaded):
        errors = validate_binding_spec(candidate)
        assert len(errors) == 1
        assert errors[0].startswith(expected_errors[0])
        assert validate_binding_spec(candidate) == errors
    assert asdict(bad) == snapshot
    assert asdict(loaded) == loaded_snapshot
    assert path.read_bytes() == source
    repaired = _write_binding(valid, path)
    assert validate_binding_spec(repaired) == []
    execution = ChannelRuntimeExecutor.from_spec(repaired).execute(
        [LayerState(R=0.8, psi=0.1), LayerState(R=0.6, psi=0.2)]
    )
    assert [layer.R for layer in execution.layers] == [0.8, 0.6]
    assert [item.evidence_source for item in execution.evidence] == [
        "current_tick",
        "current_tick",
    ]


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        ([], "channel_group 'review': channels must not be empty"),
        (
            ["P", "Unknown"],
            "channel_group 'review': references unknown channel 'Unknown'",
        ),
    ],
)
def test_channel_group_refusal_and_repaired_membership(
    members: list[str],
    expected: str,
    tmp_path: Path,
) -> None:
    """Keep malformed group declarations intact and publish repaired memberships."""
    bad = replace(_binding(), channel_groups={"review": ChannelGroupSpec(members)})
    snapshot = asdict(bad)
    path = tmp_path / "binding.json"
    loaded = _write_binding(bad, path)
    source = path.read_bytes()
    assert validate_binding_spec(bad) == [expected]
    assert validate_binding_spec(loaded) == [expected]
    assert asdict(bad) == snapshot
    assert path.read_bytes() == source
    repaired = _write_binding(
        replace(bad, channel_groups={"review": ChannelGroupSpec(["P", "Risk"])}),
        path,
    )
    assert validate_binding_spec(repaired) == []
    report = build_channel_algebra_report(repaired)
    assert report.channel_groups["review"] == ("P", "Risk")
    assert report.channel_membership["P"] == ("review",)
    assert report.channel_membership["Risk"] == ("review",)


@pytest.mark.parametrize(
    ("coupling", "expected"),
    [
        (
            CrossChannelCouplingSpec("Unknown", "Risk", 0.2),
            "source references unknown channel 'Unknown'",
        ),
        (
            CrossChannelCouplingSpec("P", "Unknown", 0.2),
            "target references unknown channel 'Unknown'",
        ),
        (CrossChannelCouplingSpec("P", "P", 0.2), "source and target must differ"),
        (
            CrossChannelCouplingSpec("P", "Risk", -0.2),
            ".strength must be finite and >= 0",
        ),
        (
            CrossChannelCouplingSpec("P", "Risk", 0.2, "feedback"),
            ".mode must be one of",
        ),
    ],
)
def test_cross_channel_coupling_refusal_and_repaired_audit_edge(
    coupling: CrossChannelCouplingSpec,
    expected: str,
    tmp_path: Path,
) -> None:
    """Refuse inconsistent edges without mutation, then emit the corrected edge."""
    bad = replace(_binding(), cross_channel_couplings=[coupling])
    snapshot = asdict(bad)
    path = tmp_path / "binding.json"
    loaded = _write_binding(bad, path)
    source = path.read_bytes()
    errors = validate_binding_spec(bad)
    assert len(errors) == 1
    assert expected in errors[0]
    assert errors[0].startswith("cross_channel_couplings[0]")
    assert validate_binding_spec(loaded) == errors
    assert asdict(bad) == snapshot
    assert path.read_bytes() == source
    repaired = _write_binding(
        replace(
            bad,
            cross_channel_couplings=[
                CrossChannelCouplingSpec("P", "Risk", 0.2, "directed"),
            ],
        ),
        path,
    )
    assert validate_binding_spec(repaired) == []
    report = build_channel_algebra_report(repaired)
    assert len(report.coupling_edges) == 1
    assert report.coupling_edges[0].to_audit_record() == {
        "source": "P",
        "target": "Risk",
        "strength": 0.2,
        "mode": "directed",
        "template": None,
    }


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_typed_coupling_refuses_without_changing_source(
    strength: float,
) -> None:
    """Reject nonfinite typed strengths before they become an audit edge."""
    coupling = CrossChannelCouplingSpec("P", "Risk", strength)
    bad = replace(_binding(), cross_channel_couplings=[coupling])
    assert validate_binding_spec(bad) == [
        "cross_channel_couplings[0].strength must be finite and >= 0"
    ]
    assert bad.cross_channel_couplings[0] is coupling
    repaired = replace(
        bad,
        cross_channel_couplings=[
            CrossChannelCouplingSpec("P", "Risk", 0.0),
        ],
    )
    assert validate_binding_spec(repaired) == []
    assert build_channel_algebra_report(repaired).coupling_edges[0].strength == 0.0


@pytest.mark.parametrize(
    "mode",
    ["bidirectional", "directed", "excitatory", "inhibitory"],
)
def test_valid_derived_channel_and_coupling_survive_file_and_audit(
    mode: str,
    tmp_path: Path,
) -> None:
    """Accept declared derivation and every coupling mode without computing a claim."""
    valid = replace(
        _binding(),
        channels={
            "Risk": ChannelSpec(
                role="risk",
                replay_semantics="derived",
                derived_from=["P", "I"],
                derive_rule="risk = phase(P)",
            )
        },
        channel_groups={"review": ChannelGroupSpec(["P", "I", "Risk"])},
        cross_channel_couplings=[CrossChannelCouplingSpec("P", "Risk", 0.0, mode)],
    )
    loaded = _write_binding(valid, tmp_path / "binding.json")
    assert validate_binding_spec(valid) == validate_binding_spec(loaded) == []
    report = build_channel_algebra_report(loaded)
    assert report.derived_channels == ("Risk",)
    assert report.missing_required_channels == ()
    assert report.channel_groups["review"] == ("P", "I", "Risk")
    assert report.coupling_edges[0].mode == mode
    assert report.coupling_edges[0].strength == 0.0
    assert json.loads(json.dumps(report.to_audit_record()))["derived_channels"] == [
        "Risk"
    ]


def test_independent_channel_errors_accumulate_in_validation_order(
    tmp_path: Path,
) -> None:
    """Return all declaration, group and coupling faults rather than fail fast."""
    bad = replace(
        _binding(),
        channels={
            "Risk": ChannelSpec(
                role="risk",
                replay_semantics="derived",
                derived_from=["P"],
            )
        },
        channel_groups={"review": ChannelGroupSpec([])},
        cross_channel_couplings=[CrossChannelCouplingSpec("P", "P", -0.2, "feedback")],
    )
    snapshot = asdict(bad)
    path = tmp_path / "binding.json"
    loaded = _write_binding(bad, path)
    errors = validate_binding_spec(bad)
    assert len(errors) == 5
    assert errors[:4] == [
        "channel 'Risk': derive_rule is required when derived_from is set",
        "channel_group 'review': channels must not be empty",
        "cross_channel_couplings[0]: source and target must differ",
        "cross_channel_couplings[0].strength must be finite and >= 0",
    ]
    assert errors[4].startswith("cross_channel_couplings[0].mode must be one of")
    assert validate_binding_spec(loaded) == errors
    assert asdict(bad) == snapshot
    assert validate_binding_spec(_write_binding(_binding(), path)) == []


@pytest.mark.parametrize("evidence_source", ["family", "driver"])
def test_required_channel_uses_declared_family_or_driver_evidence(
    evidence_source: str,
    tmp_path: Path,
) -> None:
    """Admit real configured evidence without requiring a derived-channel rule."""
    valid = replace(_binding(), channels={"Risk": ChannelSpec(role="risk")})
    if evidence_source == "family":
        valid = replace(
            valid,
            oscillator_families={
                **valid.oscillator_families,
                "risk": OscillatorFamily("Risk", "event", {}),
            },
            layers=[replace(valid.layers[0], family="risk"), valid.layers[1]],
        )
    else:
        valid = replace(
            valid, drivers=replace(valid.drivers, extra={"Risk": {"zeta": 0.02}})
        )
    loaded = _write_binding(valid, tmp_path / "binding.json")
    assert validate_binding_spec(valid) == validate_binding_spec(loaded) == []
    report = build_channel_algebra_report(loaded)
    assert report.required_channels == ("Risk",)
    assert "Risk" in report.runtime_evidence_channels
    assert report.missing_required_channels == ()
    assert report.derived_channels == ()
    execution = ChannelRuntimeExecutor.from_spec(loaded).execute(
        [LayerState(R=0.8, psi=0.1), LayerState(R=0.6, psi=0.2)]
    )
    assert [layer.R for layer in execution.layers] == [0.8, 0.6]
    assert execution.evidence[0].channel == (
        "Risk" if evidence_source == "family" else "P"
    )


@pytest.mark.parametrize("channel", ["P", "I", "S", "Risk"])
def test_empty_driver_does_not_supply_required_channel_evidence(
    channel: str,
    tmp_path: Path,
) -> None:
    """Match admission to the public report's nonempty-driver evidence rule."""
    base = _binding()
    bad = replace(
        base,
        oscillator_families={},
        channels={channel: ChannelSpec(role="required")},
        drivers=replace(
            base.drivers,
            physical={},
            informational={},
            symbolic={},
            extra={channel: {}},
        ),
    )
    snapshot = asdict(bad)
    path = tmp_path / "binding.json"
    loaded = _write_binding(bad, path)
    source = path.read_bytes()
    expected = [
        f"channel '{channel}': required channel must be backed by "
        "an oscillator family or driver"
    ]
    assert validate_binding_spec(bad) == validate_binding_spec(loaded) == expected
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.output.strip() == f"ERROR: {expected[0]}"
    assert path.read_bytes() == source
    assert build_channel_algebra_report(loaded).missing_required_channels == (channel,)
    assert asdict(bad) == snapshot
    repaired = replace(
        bad,
        drivers=replace(
            bad.drivers,
            extra={channel: {"zeta": 0.02}},
        ),
    )
    assert validate_binding_spec(repaired) == []
    assert build_channel_algebra_report(repaired).missing_required_channels == ()
    _write_binding(repaired, path)
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Valid\n")
