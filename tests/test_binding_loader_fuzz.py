# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding loader fuzz tests

from __future__ import annotations

from typing import Any

import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec

_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-10, max_value=10)
    | st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )
    | st.text(min_size=0, max_size=12)
)

_YAML_VALUE = st.recursive(
    _SCALAR,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(min_size=0, max_size=12), children, max_size=4)
    ),
    max_leaves=20,
)


def _write_yaml(tmp_path, payload: Any) -> str:
    path = tmp_path / "fuzz_binding.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def _assert_load_or_controlled_error(tmp_path, payload: Any) -> None:
    path = _write_yaml(tmp_path, payload)
    try:
        spec = load_binding_spec(path)
    except BindingLoadError as exc:
        message = str(exc)
        assert str(tmp_path) not in message
        assert "fuzz_binding.yaml" not in message or "/" not in message
        return

    errors = validate_binding_spec(spec)
    assert isinstance(errors, list)
    assert all(isinstance(error, str) for error in errors)


@given(_YAML_VALUE)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_random_yaml_payloads_fail_as_binding_load_errors(tmp_path, payload):
    _assert_load_or_controlled_error(tmp_path, payload)


@given(
    name=_YAML_VALUE,
    version=_YAML_VALUE,
    safety_tier=_YAML_VALUE,
    sample_period_s=_YAML_VALUE,
    control_period_s=_YAML_VALUE,
    layers=_YAML_VALUE,
    oscillator_families=_YAML_VALUE,
    coupling=_YAML_VALUE,
    drivers=_YAML_VALUE,
    objectives=_YAML_VALUE,
    boundaries=_YAML_VALUE,
    actuators=_YAML_VALUE,
    imprint_model=_YAML_VALUE,
    geometry_prior=_YAML_VALUE,
    protocol_net=_YAML_VALUE,
    amplitude=_YAML_VALUE,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_binding_shaped_yaml_payloads_do_not_escape_loader_boundary(
    tmp_path,
    name,
    version,
    safety_tier,
    sample_period_s,
    control_period_s,
    layers,
    oscillator_families,
    coupling,
    drivers,
    objectives,
    boundaries,
    actuators,
    imprint_model,
    geometry_prior,
    protocol_net,
    amplitude,
):
    payload = {
        "name": name,
        "version": version,
        "safety_tier": safety_tier,
        "sample_period_s": sample_period_s,
        "control_period_s": control_period_s,
        "layers": layers,
        "oscillator_families": oscillator_families,
        "coupling": coupling,
        "drivers": drivers,
        "objectives": objectives,
        "boundaries": boundaries,
        "actuators": actuators,
        "imprint_model": imprint_model,
        "geometry_prior": geometry_prior,
        "protocol_net": protocol_net,
        "amplitude": amplitude,
    }
    _assert_load_or_controlled_error(tmp_path, payload)


# Pipeline wiring: fuzzed YAML enters at the same load_binding_spec()
# boundary used by CLI validation and simulation startup.
