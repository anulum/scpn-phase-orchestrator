# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — U1 constructor validation batch tests

from __future__ import annotations

import pytest
import numpy as np
from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import CyberneticClosure
from scpn_phase_orchestrator.ssgf.tcbo import TCBOObserver
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet, Place, Transition
from scpn_phase_orchestrator.binding.types import BoundaryDef


def test_u1_action_projector_rejects_non_finite_rate_limit() -> None:
    with pytest.raises(ValueError, match="finite >= 0"):
        ActionProjector(rate_limits={"K": float("nan")}, value_bounds={"K": (0.0, 1.0)})


def test_u1_action_projector_rejects_non_finite_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        projector.project(action, float("inf"))


def test_u1_action_projector_rejects_nan_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        projector.project(action, float("nan"))


def test_u1_action_projector_rejects_boolean_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(TypeError, match="finite real scalar"):
        projector.project(action, True)  # type: ignore[arg-type]


def test_u1_action_projector_rejects_non_action_payload() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    with pytest.raises(TypeError, match="ControlAction"):
        projector.project(object(), 0.0)  # type: ignore[arg-type]


def test_u1_action_projector_rejects_non_finite_action_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=float("nan"),
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="action.value must be finite real"):
        projector.project(action, 0.0)


def test_u1_knm_template_set_rejects_non_square_template() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="square"):
        reg.add(
            KnmTemplate(
                name="bad",
                knm=np.ones((2, 3), dtype=float),
                alpha=np.ones((2, 3), dtype=float),
                description="invalid",
            )
        )


def test_u1_knm_template_set_rejects_blank_description() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="description"):
        reg.add(
            KnmTemplate(
                name="k",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.ones((2, 2), dtype=float),
                description="",
            )
        )


def test_u1_knm_template_set_rejects_non_float_dtype() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="floating-point dtypes"):
        reg.add(
            KnmTemplate(
                name="bad_dtype",
                knm=np.ones((2, 2), dtype=object),
                alpha=np.ones((2, 2), dtype=object),
                description="invalid",
            )
        )


def test_u1_knm_template_set_get_rejects_blank_name() -> None:
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("")


def test_u1_knm_template_set_get_rejects_whitespace_name() -> None:
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("   ")


def test_u1_knm_template_set_get_strips_lookup_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name="k",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.get(" k ") is tpl


def test_u1_knm_template_set_add_strips_storage_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name=" k ",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.list_names() == ["k"]


def test_u1_knm_template_set_add_strips_tabbed_storage_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name="\tk\t",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.list_names() == ["k"]


def test_u1_knm_template_set_stores_canonical_template_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name=" k ",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.get("k").name == "k"


def test_u1_boundary_observer_rejects_inverted_bounds() -> None:
    with pytest.raises(TypeError, match="BoundaryDef"):
        BoundaryObserver([object()])  # type: ignore[list-item]


def test_u1_boundary_observer_rejects_boolean_lower_bound() -> None:
    with pytest.raises(Exception, match="must be < upper"):
        BoundaryObserver(
            [
                BoundaryDef(
                    name="n",
                    variable="x",
                    lower=True,  # type: ignore[arg-type]
                    upper=1.0,
                    severity="soft",
                )
            ]
        )


def test_u1_boundary_observer_observe_rejects_negative_step() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="non-negative integer"):
        obs.observe({}, step=-1)


def test_u1_boundary_observer_observe_rejects_boolean_step() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="non-negative integer"):
        obs.observe({}, step=True)  # type: ignore[arg-type]


def test_u1_boundary_observer_observe_rejects_bool_value() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="finite float"):
        obs.observe({"R": True})  # type: ignore[dict-item]


def test_u1_boundary_observer_observe_rejects_non_numeric_value() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="finite float"):
        obs.observe({"R": "bad"})  # type: ignore[dict-item]


def test_u1_boundary_observer_observe_rejects_non_dict_values() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="dict\\[str, float\\]"):
        obs.observe([("R", 0.1)])  # type: ignore[arg-type]


def test_u1_boundary_observer_set_event_bus_rejects_wrong_type() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="EventBus"):
        obs.set_event_bus(object())  # type: ignore[arg-type]


def test_u1_boundary_observer_set_event_bus_rejects_none() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="EventBus"):
        obs.set_event_bus(None)  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="seed must be int or None"):
        GeometryCarrier(4, seed=True)  # type: ignore[arg-type]


def test_u1_geometry_carrier_reset_rejects_boolean_seed() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(TypeError, match="seed must be int or None"):
        carrier.reset(seed=True)  # type: ignore[arg-type]


def test_u1_geometry_carrier_decode_rejects_non_array_input() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        carrier.decode(z=[0.1, 0.2, 0.3, 0.4])  # type: ignore[arg-type]


def test_u1_geometry_carrier_update_rejects_boolean_epsilon() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(ValueError, match="finite positive real"):
        carrier.update(cost=0.0, epsilon=True)  # type: ignore[arg-type]


def test_u1_audit_logger_rejects_directory_path(tmp_path) -> None:
    with pytest.raises(Exception, match="directory"):
        AuditLogger(tmp_path)


def test_u1_audit_logger_rejects_blank_event_stream_path(tmp_path) -> None:
    with pytest.raises(Exception, match="event_stream path must be non-empty"):
        AuditLogger(tmp_path / "audit.jsonl", event_stream=" ")


def test_u1_audit_logger_rejects_non_path_type() -> None:
    with pytest.raises(Exception, match="str or Path"):
        AuditLogger(123)  # type: ignore[arg-type]


def test_u1_audit_logger_rejects_non_path_event_stream_type(tmp_path) -> None:
    with pytest.raises(Exception, match="str, Path, or None"):
        AuditLogger(tmp_path / "audit.jsonl", event_stream=123)  # type: ignore[arg-type]


def test_u1_audit_logger_header_rejects_non_positive_dt(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="finite positive real"):
            logger.log_header(n_oscillators=4, dt=0.0)
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_blank_method(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-empty string"):
            logger.log_header(n_oscillators=4, dt=0.1, method=" ")
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_boolean_seed(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="seed must be integer or None"):
            logger.log_header(n_oscillators=4, dt=0.1, seed=True)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_dict_binding_config(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="binding_config"):
            logger.log_header(n_oscillators=4, dt=0.1, binding_config=["bad"])  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_bool_amplitude_mode(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="amplitude_mode must be bool"):
            logger.log_header(n_oscillators=4, dt=0.1, amplitude_mode=1)  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_header_rejects_non_dict_binding_summary(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="binding_summary"):
            logger.log_header(
                n_oscillators=4,
                dt=0.1,
                binding_summary=["bad"],  # type: ignore[arg-type]
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_negative_step(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-negative integer"):
            logger.log_step(
                -1,
                object(),  # type: ignore[arg-type]
                [],
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_boolean_step(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="non-negative integer"):
            logger.log_step(
                True,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                [],
            )
    finally:
        logger._fh.close()


def test_u1_petri_adapter_step_rejects_blank_ctx_metric_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="metric names must be non-empty strings"):
        adapter.step({" ": 0.1})  # type: ignore[dict-item]


def test_u1_petri_adapter_step_rejects_boolean_ctx_metric_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="must be finite real"):
        adapter.step({"metric": True})  # type: ignore[dict-item]


def test_u1_petri_adapter_step_rejects_non_finite_ctx_metric_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="must be finite real"):
        adapter.step({"metric": float("nan")})


def test_u1_petri_adapter_rejects_non_string_regime_mapping_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="must be non-empty string"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking({"nominal": 1}),
            place_to_regime={"nominal": True},  # type: ignore[dict-item]
        )


def test_u1_audit_logger_log_step_rejects_non_upde_state(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(Exception, match="UPDEState"):
            logger.log_step(0, object(), [])
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_action_entries(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="actions\\[0\\]"):
            logger.log_step(0, state, [object()])  # type: ignore[list-item]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_list_actions(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="actions must be list"):
            logger.log_step(0, state, {})  # type: ignore[arg-type]
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_dict_channel_runtime(tmp_path) -> None:
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="channel_runtime"):
            logger.log_step(
                0,
                state,
                [],
                channel_runtime=["bad"],  # type: ignore[arg-type]
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_phases_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="phases must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, np.nan], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_omegas_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="omegas must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, np.inf], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_knm_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="knm must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.array([[1.0, 0.0], [0.0, np.nan]], dtype=float),
                alpha=np.zeros((2, 2), dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_audit_logger_log_step_rejects_non_finite_alpha_payload(tmp_path) -> None:
    from scpn_phase_orchestrator.actuation.mapper import ControlAction
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        state = UPDEState(
            regime_id="nominal",
            stability_proxy=0.0,
            layers=[LayerState(0.0, 0.0)],
            cross_layer_alignment=[],
        )
        with pytest.raises(Exception, match="alpha must contain only finite values"):
            logger.log_step(
                0,
                state,
                [ControlAction("K", "global", 0.1, 1.0, "u1-test")],
                phases=np.array([0.0, 0.1], dtype=float),
                omegas=np.array([1.0, 1.0], dtype=float),
                knm=np.ones((2, 2), dtype=float),
                alpha=np.array([[0.0, 0.0], [0.0, np.inf]], dtype=float),
            )
    finally:
        logger._fh.close()


def test_u1_geometry_carrier_rejects_non_positive_latent_dim() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GeometryCarrier(n_oscillators=4, z_dim=0, lr=0.1)


def test_u1_geometry_carrier_update_rejects_non_positive_epsilon() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="finite positive real"):
        carrier.update(cost=0.1, epsilon=0.0)


def test_u1_geometry_carrier_update_rejects_non_callable_cost_fn() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(TypeError, match="callable or None"):
        carrier.update(cost=0.1, cost_fn=1.0)  # type: ignore[arg-type]


def test_u1_geometry_carrier_decode_rejects_wrong_shape() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="length 2"):
        carrier.decode(np.zeros((2, 1), dtype=float))


def test_u1_geometry_carrier_reset_rejects_non_integer_seed() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(TypeError, match="int or None"):
        carrier.reset(seed="bad")  # type: ignore[arg-type]


def test_u1_cybernetic_closure_rejects_negative_max_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        CyberneticClosure(carrier=carrier, max_steps=-1)


def test_u1_cybernetic_closure_run_rejects_negative_outer_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="non-negative integer"):
        closure.run(np.zeros(4, dtype=float), -1)


def test_u1_cybernetic_closure_run_rejects_non_vector_phases() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="1D vector"):
        closure.run(np.zeros((2, 2), dtype=float), 0)


def test_u1_cybernetic_closure_step_rejects_mismatched_phase_length() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="oscillator count"):
        closure.step(np.zeros(3, dtype=float))


def test_u1_tcbo_observer_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        TCBOObserver(tau_h1=1.5)


def test_u1_tcbo_observer_rejects_boolean_tau() -> None:
    with pytest.raises(TypeError, match="finite real values"):
        TCBOObserver(tau_h1=True)  # type: ignore[arg-type]


def test_u1_tcbo_observer_observe_rejects_non_finite_phase() -> None:
    obs = TCBOObserver()
    with pytest.raises(ValueError, match="finite values"):
        obs.observe(np.array([0.0, np.nan], dtype=float))


def test_u1_petri_adapter_rejects_non_string_regime_mapping_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    with pytest.raises(Exception, match="non-empty string"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking(tokens={"nominal": 1}),
            place_to_regime={"nominal": 1},  # type: ignore[dict-item]
        )


def test_u1_petri_adapter_rejects_empty_mapping() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    with pytest.raises(Exception, match="must not be empty"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking(tokens={"nominal": 1}),
            place_to_regime={},
        )


def test_u1_petri_adapter_step_rejects_whitespace_ctx_metric() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "nominal"},
    )
    with pytest.raises(Exception, match="non-empty strings"):
        adapter.step({" ": 0.0})


def test_u1_petri_adapter_step_rejects_boolean_ctx_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "nominal"},
    )
    with pytest.raises(Exception, match="finite real"):
        adapter.step({"stability_proxy": True})  # type: ignore[dict-item]


def test_u1_petri_adapter_accepts_whitespace_wrapped_regime_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": " nominal "},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"


def test_u1_petri_adapter_accepts_whitespace_wrapped_place_key() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={" nominal ": "nominal"},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"


def test_u1_petri_adapter_accepts_uppercase_regime_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"
