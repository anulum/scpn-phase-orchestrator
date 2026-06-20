# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio display-table builders

"""Layer and oscillator display-table builders from a binding spec."""

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import BindingSpec


def build_layer_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return editable layer rows for the Studio oscillator canvas.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    tuple[dict[str, object], ...]
        Editable layer rows for the Studio oscillator canvas.
    """
    return tuple(
        {
            "index": int(layer.index),
            "name": layer.name,
            "oscillator_count": len(layer.oscillator_ids),
            "family": layer.family or "",
            "omega_count": len(layer.omegas or ()),
        }
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )


def build_oscillator_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return oscillator rows suitable for Streamlit data editing.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    tuple[dict[str, object], ...]
        Oscillator rows suitable for Streamlit data editing.
    """
    family_channels = {
        family_name: family.channel
        for family_name, family in spec.oscillator_families.items()
    }
    rows: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        channel = family_channels.get(layer.family or "", "")
        for oscillator_id in layer.oscillator_ids:
            rows.append(
                {
                    "layer": layer.name,
                    "layer_index": int(layer.index),
                    "oscillator_id": oscillator_id,
                    "family": layer.family or "",
                    "channel": channel,
                }
            )
    return tuple(rows)
