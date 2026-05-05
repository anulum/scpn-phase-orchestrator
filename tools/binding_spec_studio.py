#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# Copyright © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Binding-spec studio for quickly editing and validating binding specs."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import streamlit as st  # type: ignore[import-not-found]

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.resolved import (
    format_resolved_binding_config,
    resolved_binding_config,
)
from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
    OscillatorFamily,
    resolve_extractor_type,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

DEFAULT_PREVIEW_CHANNELS = {"P", "I", "S"}
VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")
VALID_EXTRACTOR_PREVIEW = {
    "hilbert",
    "wavelet",
    "zero_crossing",
    "event",
    "ring",
    "graph",
}
PREVIEW_SIGNAL = np.linspace(0.0, 2.0, 160)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_text_from_upload(uploaded_bytes: bytes | None) -> str:
    if not uploaded_bytes:
        return ""
    return uploaded_bytes.decode("utf-8", errors="replace")


def _load_binding_from_text(
    yaml_text: str,
) -> tuple[BindingSpec | None, list[str]]:
    if not yaml_text.strip():
        return None, []
    with NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_text)
        f.flush()
        tmp_path = Path(f.name)
    try:
        try:
            spec = load_binding_spec(tmp_path)
        except BindingLoadError as exc:
            return None, [str(exc)]
    finally:
        tmp_path.unlink()

    errors = validate_binding_spec(spec)
    return spec, errors


def _render_mapping_table(spec: BindingSpec) -> None:
    family_rows = []
    for name, family in spec.oscillator_families.items():
        family_rows.append(
            {
                "family": name,
                "channel": family.channel,
                "extractor": family.extractor_type,
                "driver_config": bool(family.config),
                "nodes": ", ".join(
                    str(value) for value in (family.config or {}).get("channel_ids", [])
                ),
            }
        )
    if family_rows:
        st.dataframe(family_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No oscillator families found in this spec.")

    channel_rows = []
    for channel in sorted(DEFAULT_PREVIEW_CHANNELS):
        cfg = spec.drivers.channel_config(channel)
        channel_rows.append(
            {
                "channel": channel,
                "driver": "present" if cfg else "empty",
                "keys": ", ".join(sorted(cfg.keys())),
            }
        )
    st.markdown("### P/I/S driver assignments")
    st.table(channel_rows)

    custom_channels = []
    for channel in sorted(spec.channels):
        channel_spec = spec.channels[channel]
        if channel in DEFAULT_PREVIEW_CHANNELS:
            continue
        custom_channels.append(
            {
                "channel": channel,
                "role": channel_spec.role,
                "required": channel_spec.required,
                "coupling": channel_spec.coupling_participation,
                "visibility": channel_spec.supervisor_visibility,
                "replay_semantics": channel_spec.replay_semantics,
            }
        )
    if custom_channels:
        st.markdown("### Extra declared channels")
        st.dataframe(custom_channels, use_container_width=True, hide_index=True)


def _extractor_preview(
    family_name: str,
    family: OscillatorFamily,
    sample_rate: float,
) -> dict[str, str]:
    extractor = resolve_extractor_type(family.extractor_type)
    if extractor not in VALID_EXTRACTOR_PREVIEW:
        return {
            "family": family_name,
            "extractor": extractor,
            "preview_status": "unsupported extractor",
            "theta": "—",
            "omega": "—",
            "quality": "—",
        }

    try:
        if extractor in {"hilbert", "wavelet", "zero_crossing"}:
            cfg = family.config or {}
            freq = float(cfg.get("freq_hz", 1.0))
            signal = np.sin(2 * np.pi * freq * PREVIEW_SIGNAL)
            signal += 0.02 * np.arange(len(signal))
            state = PhysicalExtractor(node_id=f"preview-{family_name}").extract(
                signal.astype(np.float64), sample_rate=sample_rate
            )[0]
        elif extractor == "event":
            events = np.cumsum([0.05, 0.37, 0.18, 0.40], dtype=np.float64)
            events = np.cumsum(events)
            state = InformationalExtractor(node_id=f"preview-{family_name}").extract(
                events, sample_rate=sample_rate
            )[0]
        else:
            cfg = family.config or {}
            n_states = int(cfg.get("n_states", 4))
            n_states = max(2, min(n_states, 128))
            mode = str(cfg.get("mode", "ring"))
            steps = np.array([0, 1, 2, 3, 1, 2, 3, 0], dtype=np.float64)
            state = SymbolicExtractor(
                n_states=n_states,
                node_id=f"preview-{family_name}",
                mode="graph" if mode == "graph" else "ring",
            ).extract(steps, sample_rate=sample_rate)[-1]
    except Exception as exc:
        return {
            "family": family_name,
            "extractor": extractor,
            "preview_status": f"failed: {exc}",
            "theta": "—",
            "omega": "—",
            "quality": "—",
        }
    return {
        "family": family_name,
        "extractor": extractor,
        "preview_status": "ok",
        "theta": f"{state.theta:.3f}",
        "omega": f"{state.omega:.3f}",
        "quality": f"{state.quality:.3f}",
    }


def _render_preview(spec: BindingSpec) -> None:
    sample_rate = 1.0 / spec.sample_period_s if spec.sample_period_s > 0 else 100.0
    rows = []
    for name, family in sorted(spec.oscillator_families.items()):
        rows.append(_extractor_preview(name, family, sample_rate=sample_rate))
    if not rows:
        st.info("No families to preview.")
        return

    st.markdown("### Extractor preview (synthetic sample)")
    st.table(rows)


def _resolved_summary(spec: BindingSpec) -> None:
    resolved_summary = format_resolved_binding_config(resolved_binding_config(spec))
    with st.expander("Resolved runtime defaults", expanded=False):
        for line in resolved_summary:
            st.text(line)


def _domainpacks_dir() -> Path:
    candidate = Path(__file__).resolve().parent.parent / "domainpacks"
    if candidate.exists():
        return candidate
    return Path("domainpacks")


def main() -> None:
    st.set_page_config(page_title="SPO Binding Spec Studio", layout="wide")
    st.title("🧪 SPO Binding-Spec Studio")
    st.caption(
        "Create, edit, validate, preview, and scaffold domainpacks from a "
        "binding spec YAML."
    )

    domainpacks_dir = _domainpacks_dir()
    available = []
    if domainpacks_dir.exists():
        available = sorted(
            d.name
            for d in domainpacks_dir.iterdir()
            if d.is_dir() and (d / "binding_spec.yaml").exists()
        )
    if "spec_text" not in st.session_state:
        default_pack = (
            "minimal_domain"
            if "minimal_domain" in available
            else (available[0] if available else "")
        )
        default_text = ""
        if default_pack:
            default_text = _read_text_file(
                domainpacks_dir / default_pack / "binding_spec.yaml"
            )
        st.session_state["spec_text"] = default_text

    st.subheader("1) Source binding spec")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        chosen = st.selectbox("Load an existing domainpack", [""] + available, index=0)
        if st.button("Load selected spec") and chosen:
            st.session_state["spec_text"] = _read_text_file(
                domainpacks_dir / chosen / "binding_spec.yaml"
            )
            st.rerun()
    with c2:
        uploaded = st.file_uploader(
            "Or upload a binding_spec YAML", type=["yml", "yaml"]
        )
        if uploaded:
            st.session_state["spec_text"] = _load_text_from_upload(uploaded.read())
            st.rerun()

    spec_text = st.text_area(
        "binding_spec.yaml", value=st.session_state["spec_text"], height=360
    )
    st.session_state["spec_text"] = spec_text

    st.subheader("2) Validate and inspect")
    spec, errors = _load_binding_from_text(spec_text)
    if spec is None:
        for msg in errors:
            st.error(msg)
        st.stop()

    if errors:
        st.warning("Spec parses but schema validation found issues:")
        for issue in errors:
            st.error(issue)
    else:
        st.success("Spec parsed and validated successfully.")
    _resolved_summary(spec)

    st.subheader("3) Mapping")
    _render_mapping_table(spec)

    st.subheader("4) Extractor output preview")
    _render_preview(spec)

    st.subheader("5) Export binding spec")
    st.download_button(
        "Download binding_spec.yaml",
        data=spec_text,
        file_name="binding_spec.yaml",
        mime="application/x-yaml",
        disabled=bool(errors),
    )
    if errors:
        st.info("Fix validation errors before exporting the binding spec.")


if __name__ == "__main__":
    main()
