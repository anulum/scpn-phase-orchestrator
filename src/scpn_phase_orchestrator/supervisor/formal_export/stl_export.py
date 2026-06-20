# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL specification PRISM exporter

"""PRISM text exporter for supervisor STL monitor specifications."""

from __future__ import annotations

import re

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicySTLSpec,
)

from ._shared import PrismExport, _identifier, _unique_identifier

_SIMPLE_STL_RE = re.compile(r"^(always|eventually)\s*\((.*)\)\s*$")


_STL_PREDICATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|==)\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*$"
)


def _stl_mapping(specs: list[PolicySTLSpec]) -> dict[str, str]:
    used: set[str] = set()
    return {
        spec.name: _unique_identifier(spec.name, prefix="stl", used=used)
        for spec in specs
    }


def _stl_predicates(spec: PolicySTLSpec) -> tuple[str, list[tuple[str, str, float]]]:
    match = _SIMPLE_STL_RE.match(spec.spec.strip())
    if match is None:
        raise PolicyError(f"STL monitor {spec.name!r} uses unsupported export syntax")
    temporal_op = match.group(1)
    predicates: list[tuple[str, str, float]] = []
    for raw_predicate in re.split(r"\s+(?:and|&&)\s+", match.group(2)):
        predicate_match = _STL_PREDICATE_RE.match(raw_predicate)
        if predicate_match is None:
            raise PolicyError(
                f"STL monitor {spec.name!r} uses unsupported predicate syntax"
            )
        signal, op, threshold = predicate_match.groups()
        predicates.append((signal, op, float(threshold)))
    return temporal_op, predicates


def _stl_signal_mapping(
    specs: list[PolicySTLSpec],
) -> tuple[dict[str, str], dict[str, tuple[str, list[tuple[str, str, float]]]]]:
    parsed: dict[str, tuple[str, list[tuple[str, str, float]]]] = {}
    signals: set[str] = set()
    for spec in specs:
        temporal_op, predicates = _stl_predicates(spec)
        parsed[spec.name] = (temporal_op, predicates)
        signals.update(signal for signal, _op, _threshold in predicates)
    used: set[str] = set()
    signal_names = {
        signal: _unique_identifier(signal, prefix="s", used=used)
        for signal in sorted(signals)
    }
    return signal_names, parsed


def _stl_expr(
    predicates: list[tuple[str, str, float]],
    signal_names: dict[str, str],
) -> str:
    parts = [
        f"{signal_names[signal]} {op} {threshold:.17g}"
        for signal, op, threshold in predicates
    ]
    return " & ".join(parts) if parts else "true"


def export_stl_specs_prism(
    specs: list[PolicySTLSpec],
    *,
    module_name: str = "spo_stl",
) -> PrismExport:
    """Serialise policy-declared STL monitors into PRISM label surfaces.

    This export covers the builtin STL subset used by ``STLMonitor``:
    ``always (...)`` and ``eventually (...)`` over numeric predicate
    conjunctions. The model is a single-state abstraction with signal
    constants and per-monitor satisfied/violated labels for property checks.

    Parameters
    ----------
    specs : list[PolicySTLSpec]
        Policy-declared STL monitor specifications.
    module_name : str
        Name of the emitted model-checker module.

    Returns
    -------
    PrismExport
        The PRISM label-surface export of the STL monitors.

    Raises
    ------
    PolicyError
        If the STL specs violate the export policy.
    """
    if not specs:
        raise PolicyError("cannot export STL monitors without specs")
    for spec in specs:
        if not spec.name:
            raise PolicyError("STL monitor names must not be empty")
        if spec.severity not in {"soft", "hard"}:
            raise PolicyError(f"STL monitor {spec.name!r} has unsupported severity")

    stl_names = _stl_mapping(specs)
    signal_names, parsed = _stl_signal_mapping(specs)
    module_identifier = _identifier(module_name, prefix="module")

    lines = [
        "mdp",
        "",
        "// Generated from SCPN policy STL monitors for PRISM model checking.",
        "// Signal constants represent one sampled trace point or scenario bound.",
    ]
    if signal_names:
        lines.append("// STL signal constants:")
        lines.extend(f"//   {raw} -> {mapped}" for raw, mapped in signal_names.items())
        lines.extend(f"const double {mapped};" for mapped in signal_names.values())
    lines.extend(
        [
            "",
            f"module {module_identifier}",
            "  state : [0..0] init 0;",
            "endmodule",
            "",
        ]
    )

    for spec in specs:
        stl_id = stl_names[spec.name]
        temporal_op, predicates = parsed[spec.name]
        expr = _stl_expr(predicates, signal_names)
        lines.append(
            f"// STL {spec.name!r}: {temporal_op} monitor, severity={spec.severity}"
        )
        lines.append(f'label "stl_{stl_id}_satisfied" = {expr};')
        lines.append(f'label "stl_{stl_id}_violated" = !({expr});')

    return PrismExport(
        model="\n".join(lines) + "\n",
        place_names={},
        metric_names=signal_names,
        transition_names={},
        stl_names=stl_names,
    )
