#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy Studio
#
# Interactive policy authoring and validation with structured rule builder,
# autocomplete, cooldown/rate-limit preview, and audit dry-run diagnostics.

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import streamlit as st  # type: ignore[import-not-found]
import yaml

from scpn_phase_orchestrator.binding import (
    BindingLoadError,
    load_binding_spec,
    validate_binding_spec,
)
from scpn_phase_orchestrator.supervisor.policy_diagnostics import (
    PolicyDryRunReport,
    dry_run_policy_rules,
)
from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

KNOWN_REGIMES = ["NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"]
KNOWN_METRICS = [
    "R",
    "R_good",
    "R_bad",
    "stability_proxy",
    "pac_max",
    "mean_amplitude",
    "subcritical_fraction",
    "amplitude_spread",
    "mean_amplitude_layer",
    "boundary_violation_count",
    "imprint_mean",
]
LAYER_METRICS = {
    "R",
    "R_good",
    "R_bad",
    "amplitude_spread",
    "mean_amplitude_layer",
}
KNOWN_OPERATORS = [">", ">=", "<", "<=", "=="]
KNOWN_KNOBS = ["K", "alpha", "zeta", "Psi"]
DEFAULT_SCOPE_COUNT = 3
MAX_PREVIEW_STEPS = 10_000


def _available_domainpacks() -> list[str]:
    """Return all domainpacks that contain a policy.yaml file."""
    root = Path(__file__).resolve().parent.parent / "domainpacks"
    if not root.exists():
        root = Path("domainpacks")
    if not root.exists():
        return []
    packs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "policy.yaml").exists():
            packs.append(child.name)
    return packs


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_upload(uploaded: Any) -> str:
    if uploaded is None:
        return ""
    return uploaded.read().decode("utf-8", errors="replace")


def _load_policy_yaml(policy_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse policy YAML and ensure dictionary root shape."""
    if not policy_text.strip():
        return {"rules": []}, []
    try:
        raw = yaml.safe_load(policy_text)
    except yaml.YAMLError as exc:  # pragma: no cover - delegated to YAML parser tests
        return None, [f"YAML parse error: {exc}"]

    if raw is None:
        return {"rules": []}, []
    if not isinstance(raw, dict):
        return (
            None,
            [
                "Schema diagnostic: policy YAML root must be a mapping with "
                "optional 'rules'.",
            ],
        )
    return raw, []


def _validate_policy_yaml(
    policy_text: str,
) -> tuple[dict[str, Any], list[str], list[Any] | None]:
    """Parse and validate policy rules; return raw payload and diagnostics."""
    payload, yaml_errors = _load_policy_yaml(policy_text)
    if payload is None:
        return {}, yaml_errors, None
    with NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as tmp:
        yaml.dump(payload, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    try:
        rules = load_policy_rules(tmp_path)
    except ValueError as exc:
        return payload, [str(exc)], None
    finally:
        tmp_path.unlink()

    return payload, [], rules


def _load_binding_text(binding_text: str) -> tuple[Any | None, list[str]]:
    if not binding_text.strip():
        return None, []
    with NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as tmp:
        tmp.write(binding_text)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        spec = load_binding_spec(tmp_path)
    except (BindingLoadError, OSError, ValueError) as exc:
        return None, [str(exc)]
    finally:
        tmp_path.unlink()

    errors = validate_binding_spec(spec)
    if errors:
        return spec, errors
    return spec, []


def _binding_layer_count(binding_text: str) -> int:
    spec, errors = _load_binding_text(binding_text)
    if errors or spec is None:
        return 0
    return len(getattr(spec, "layers", ()))


def _scope_options(scope_count: int) -> list[str]:
    return ["global"] + [f"layer_{idx}" for idx in range(scope_count)]


def _to_yaml_text(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def _metric_uses_layer(metric: str) -> bool:
    return metric in LAYER_METRICS


def _build_rule_dict(
    name: str,
    regimes: list[str],
    logic: str,
    cooldown_s: float,
    max_fires: int,
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "regime": regimes,
        "conditions": conditions,
        "logic": logic,
        "actions": actions,
        "cooldown_s": cooldown_s,
        "max_fires": max_fires,
    }


def _action_chain_preview(
    payload: dict[str, Any],
    horizon_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    for idx, rule in enumerate(payload.get("rules", [])):
        if not isinstance(rule, dict):
            continue
        name = rule.get("name", f"rule_{idx + 1}")
        cooldown = float(rule.get("cooldown_s", 0.0))
        max_fires = int(rule.get("max_fires", 0))
        if horizon_steps <= 0:
            rows.append(
                {
                    "name": name,
                    "cooldown_s": cooldown,
                    "max_fires": max_fires,
                    "rate_limit_notes": "No horizon selected",
                }
            )
            continue

        if cooldown <= 0:
            max_by_cooldown = horizon_steps
        else:
            tick_window = max(1, int(math.ceil(cooldown)))
            max_by_cooldown = math.ceil(horizon_steps / tick_window)
        effective_max = max_by_cooldown
        if max_fires > 0:
            effective_max = min(effective_max, max_fires)

        notes = (
            f"At most {effective_max:.0f} fires over {horizon_steps} steps"
            if cooldown > 0 or max_fires > 0
            else "Unbounded (no cooldown/max_fires constraints)"
        )

        if cooldown > 0 and max_fires > 0:
            notes += f" (cooldown floor {tick_window} steps, max fires {max_fires})"
        elif cooldown > 0:
            notes += f" (cooldown floor {tick_window} steps)"
        elif max_fires > 0:
            notes += " (hard cap from max_fires)"

        rows.append(
            {
                "name": name,
                "cooldown_s": cooldown,
                "max_fires": max_fires,
                "rate_limit_notes": notes,
            }
        )
    return rows


def _parse_audit_rows(raw_bytes: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    if not raw_bytes:
        return [], ["Upload a JSONL audit file for dry-run evaluation."]
    lines = []
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return [], [f"Line {idx}: JSON parse error ({exc})"]
    return lines, []


def _render_rule_summary(rules: list[dict[str, Any]]) -> None:
    if not rules:
        st.info("No rules in this policy yet.")
        return

    rows = []
    for idx, rule in enumerate(rules):
        conditions = rule.get("conditions") or []
        actions = rule.get("actions") or []
        cond_count = len(conditions)
        action_count = len(actions)
        rows.append(
            {
                "rule": rule.get("name", f"rule_{idx + 1}"),
                "regimes": ", ".join(rule.get("regime", [])),
                "conditions": cond_count,
                "actions": action_count,
                "cooldown_s": rule.get("cooldown_s", 0),
                "max_fires": rule.get("max_fires", 0),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_diagnostic_summary(report: PolicyDryRunReport) -> None:
    st.subheader("Policy dry-run report")
    st.write(f"Steps analysed: {report.steps}")
    if report.rules:
        st.write(f"Rules loaded: {len(report.rules)}")

    if report.unreachable_rules:
        st.warning(
            "Unreachable rules: "
            + ", ".join(str(rule) for rule in report.unreachable_rules)
        )

    if report.overlapping_steps:
        st.warning(
            "Rule overlap steps: "
            + ", ".join(str(step) for step in report.overlapping_steps[:16])
            + ("..." if len(report.overlapping_steps) > 16 else "")
        )

    if report.action_collision_steps:
        st.warning(
            "Action-collision steps: "
            + ", ".join(str(step) for step in report.action_collision_steps[:16])
            + ("..." if len(report.action_collision_steps) > 16 else "")
        )

    counts = [
        {"rule": rule, "fires": report.fire_counts[rule]} for rule in report.rules
    ]
    st.markdown("#### Rule fire counts")
    if counts:
        st.dataframe(counts, use_container_width=True, hide_index=True)


def _scope_defaults(scope_count: int) -> list[str]:
    return _scope_options(max(DEFAULT_SCOPE_COUNT, scope_count))


def main() -> None:
    st.set_page_config(
        page_title="SPO Policy Studio",
        page_icon="🧭",
        layout="wide",
    )
    st.title("🧭 SPO Policy Studio")
    st.caption(
        "Interactive policy DSL authoring with diagnostics, structured rule builder, "
        "cooldown/rate-limit preview, and audit dry-run evaluation."
    )

    packs = _available_domainpacks()
    default_pack = (
        "minimal_domain" if "minimal_domain" in packs else (packs[0] if packs else "")
    )
    pack_options = [""] + packs
    default_pack_index = pack_options.index(default_pack) if default_pack else 0

    if "policy_text" not in st.session_state:
        if packs and default_pack:
            st.session_state["policy_text"] = _read_text(
                Path(__file__).resolve().parent.parent
                / "domainpacks"
                / default_pack
                / "policy.yaml"
            )
        else:
            st.session_state["policy_text"] = ""

    if "binding_text" not in st.session_state:
        st.session_state["binding_text"] = ""
    if "scope_layer_count" not in st.session_state:
        st.session_state["scope_layer_count"] = 0

    st.subheader("1) Policy source")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        pack_name = st.selectbox(
            "Load policy from domainpack",
            pack_options,
            index=default_pack_index,
        )
        if st.button("Load selected policy", key="load_pack"):
            if pack_name:
                pack_dir = (
                    Path(__file__).resolve().parent.parent / "domainpacks" / pack_name
                )
                st.session_state["policy_text"] = _read_text(pack_dir / "policy.yaml")
                st.session_state["binding_text"] = _read_text(
                    pack_dir / "binding_spec.yaml"
                )
                st.session_state["scope_layer_count"] = _binding_layer_count(
                    st.session_state["binding_text"]
                )
            else:
                st.info("Select a domainpack first.")
    with c2:
        uploaded_policy = st.file_uploader("Upload policy.yaml", type=["yaml", "yml"])
        if uploaded_policy is not None:
            st.session_state["policy_text"] = _read_upload(uploaded_policy)

    st.text_area(
        "policy.yaml",
        value=st.session_state["policy_text"],
        height=340,
        key="policy_text_area",
    )
    policy_text = st.session_state["policy_text_area"]
    st.session_state["policy_text"] = policy_text

    payload, parse_errors, rules = _validate_policy_yaml(policy_text)
    if parse_errors:
        st.error("Schema diagnostics")
        for msg in parse_errors:
            st.error(msg)
        st.stop()

    if rules:
        st.success(f"Policy validated: {len(rules)} rule(s) loaded.")
    else:
        st.info("Policy loaded with no rules yet.")

    _render_rule_summary(payload.get("rules", []))

    st.subheader("2) Structured rule builder")
    with st.expander("Add rule", expanded=True):
        existing_rules = payload.get("rules", [])
        rule_name = st.text_input(
            "Rule name",
            value=f"rule_{len(existing_rules) + 1}",
            key="rule_name",
        )
        selected_regimes = st.multiselect(
            "Regimes",
            KNOWN_REGIMES,
            default=[KNOWN_REGIMES[1]],
            key="rule_regimes",
        )
        condition_count = st.slider("Conditions", min_value=1, max_value=4, value=1)

        conditions: list[dict[str, Any]] = []
        for idx in range(condition_count):
            cols = st.columns([2, 1, 1.2, 2])
            with cols[0]:
                metric = st.selectbox(
                    f"Condition {idx + 1} metric",
                    KNOWN_METRICS,
                    key=f"cond_metric_{idx}",
                )
            with cols[1]:
                operator = st.selectbox(
                    "Operator",
                    KNOWN_OPERATORS,
                    key=f"cond_op_{idx}",
                )
            with cols[2]:
                if _metric_uses_layer(metric):
                    layer = st.number_input(
                        "Layer",
                        min_value=0,
                        step=1,
                        key=f"cond_layer_{idx}",
                    )
                else:
                    layer = None
            with cols[3]:
                threshold = st.number_input(
                    "Threshold",
                    value=0.0,
                    step=0.1,
                    key=f"cond_threshold_{idx}",
                )

            cond: dict[str, Any] = {
                "metric": metric,
                "op": operator,
                "threshold": float(threshold),
            }
            if layer is not None:
                cond["layer"] = int(layer)
            conditions.append(cond)

        actions_count = st.slider("Actions", min_value=1, max_value=4, value=1)
        scope_values = _scope_defaults(st.session_state.get("scope_layer_count", 0))
        actions: list[dict[str, Any]] = []
        for idx in range(actions_count):
            cols = st.columns([1.2, 1.2, 1.2, 1.3])
            with cols[0]:
                knob = st.selectbox(
                    f"Action {idx + 1} knob",
                    KNOWN_KNOBS,
                    key=f"act_knob_{idx}",
                )
            with cols[1]:
                scope = st.selectbox(
                    f"Action {idx + 1} scope",
                    scope_values,
                    key=f"act_scope_{idx}",
                )
            with cols[2]:
                value = st.number_input(
                    f"Action {idx + 1} value",
                    value=0.0,
                    step=0.1,
                    key=f"act_value_{idx}",
                )
            with cols[3]:
                ttl_s = st.number_input(
                    f"Action {idx + 1} ttl_s",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"act_ttl_{idx}",
                )
            actions.append(
                {
                    "knob": knob,
                    "scope": scope,
                    "value": float(value),
                    "ttl_s": float(ttl_s),
                }
            )

        cooldown_s = st.slider(
            "Cooldown (seconds)",
            min_value=0.0,
            max_value=120.0,
            value=0.0,
        )
        logic = st.radio(
            "Condition logic",
            ["AND", "OR"],
            index=0,
            horizontal=True,
        )
        max_fires = st.number_input(
            "Max fires",
            min_value=0,
            value=0,
            step=1,
            help="0 means unlimited",
        )
        append_btn = st.button("Append rule to policy YAML", type="primary")
        if append_btn:
            if not rule_name.strip():
                st.error("Rule name is required.")
            elif not selected_regimes:
                st.error("Select at least one regime.")
            elif any(rule.get("name") == rule_name for rule in existing_rules):
                st.error(f"Rule name '{rule_name}' already exists.")
            else:
                new_rule = _build_rule_dict(
                    name=rule_name.strip(),
                    regimes=selected_regimes,
                    logic=logic,
                    cooldown_s=float(cooldown_s),
                    max_fires=int(max_fires),
                    conditions=conditions,
                    actions=actions,
                )
                payload["rules"] = existing_rules + [new_rule]
                st.session_state["policy_text"] = _to_yaml_text(payload)
                st.success(f"Rule '{rule_name}' appended.")
                st.rerun()

    st.subheader("3) Cooldown and rate-limit preview")
    horizon = st.slider(
        "Preview horizon (steps)",
        min_value=10,
        max_value=1000,
        value=100,
    )
    preview_rows = _action_chain_preview(payload, min(horizon, MAX_PREVIEW_STEPS))
    if preview_rows:
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)

    st.subheader("4) Dry-run against audit log")
    st.write(
        "Upload an `audit.jsonl` file to test rule firing, overlaps, and collisions."
    )
    audit_file = st.file_uploader("Upload audit.jsonl", type=["jsonl", "json"])
    uploaded_binding = st.file_uploader(
        "Optional: upload binding_spec.yaml for dry-run context",
        type=["yaml", "yml"],
    )

    if uploaded_binding is not None:
        st.session_state["binding_text"] = _read_upload(uploaded_binding)
        st.session_state["scope_layer_count"] = _binding_layer_count(
            st.session_state["binding_text"]
        )
    elif uploaded_binding is None and st.session_state["binding_text"]:
        st.session_state["scope_layer_count"] = _binding_layer_count(
            st.session_state["binding_text"]
        )
    elif uploaded_binding is None and not st.session_state["binding_text"] and packs:
        # Keep existing default binding text from selected domainpack if available.
        st.info(
            "No binding spec uploaded. If you loaded a domainpack, "
            "that spec will be used."
        )

    run_dry = st.button("Run dry-run diagnostics", type="primary")
    if run_dry:
        if not audit_file:
            st.error("Upload an audit file first.")
            return
        binding_text = st.session_state["binding_text"].strip()
        if not binding_text:
            st.error(
                "Upload a binding spec or load a domainpack with binding_spec.yaml."
            )
            return

        spec, binding_errors = _load_binding_text(binding_text)
        if binding_errors:
            st.error("Binding spec errors:")
            for msg in binding_errors:
                st.error(msg)
            return

        entries, audit_errors = _parse_audit_rows(audit_file.read())
        if audit_errors:
            for msg in audit_errors:
                st.error(msg)
            return

        if not entries:
            st.warning("Audit has no parseable state entries.")
            return

        report = dry_run_policy_rules(
            rules,
            entries,
            good_layers=list(spec.objectives.good_layers),
            bad_layers=list(spec.objectives.bad_layers),
        )
        _render_diagnostic_summary(report)


if __name__ == "__main__":
    main()
