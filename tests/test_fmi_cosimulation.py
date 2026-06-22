# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FMI 3.0 co-simulation export tests

"""Tests for the FMI 3.0 co-simulation export of the Koopman MPC.

The slave is driven the way an FMI master would (set → step → get), its control
is checked against the controller it wraps, the generated ``modelDescription.xml``
is parsed and checked for FMI 3.0 conformance, the ``.fmu`` archive is unpacked,
and the input-validation surface is exercised in full.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from defusedxml.ElementTree import fromstring

from scpn_phase_orchestrator.actuation.koopman_mpc import (
    KoopmanMPCConfig,
    KoopmanMPCController,
)
from scpn_phase_orchestrator.adapters.fmi_cosimulation import (
    CoSimulationSlave,
    FMIVariable,
    generate_model_description,
    write_fmu,
)
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary,
    fit_koopman_predictor,
)
from scpn_phase_orchestrator.runtime.dvoc_oscillation_damping import (
    underdamped_oscillator,
)


def _controller() -> KoopmanMPCController:
    state_matrix, input_matrix = underdamped_oscillator(
        frequency_hz=0.5, damping_ratio=0.02, dt=0.02
    )
    rng = np.random.default_rng(0)
    states = rng.normal(0.0, 1.0, size=(400, 2))
    inputs = rng.normal(0.0, 1.0, size=(400, 1))
    next_states = states @ state_matrix.T + inputs @ input_matrix.T
    predictor = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=2),
    )
    config = KoopmanMPCConfig(horizon=15, input_lower=-50.0, input_upper=50.0)
    return KoopmanMPCController(predictor, config)


def _slave() -> CoSimulationSlave:
    return CoSimulationSlave(_controller(), model_name="scpn_koopman_mpc")


# --------------------------------------------------------------------------- #
# Variable layout                                                             #
# --------------------------------------------------------------------------- #
def test_slave_exposes_state_reference_and_control_variables() -> None:
    slave = _slave()
    assert slave.state_dim == 2
    assert slave.input_dim == 1
    causalities = [var.causality for var in slave.variables]
    assert causalities == ["input", "input", "input", "input", "output"]
    assert isinstance(slave.variables[0], FMIVariable)
    assert slave.variables[-1].value_reference == 1000


# --------------------------------------------------------------------------- #
# Co-simulation lifecycle reproduces the controller                           #
# --------------------------------------------------------------------------- #
def test_slave_step_matches_the_wrapped_controller() -> None:
    slave = _slave()
    slave.enter_initialization_mode()
    slave.set_float64([0, 1], [1.0, 0.0])
    slave.exit_initialization_mode()
    slave.do_step(0.0, 0.02)
    control = slave.get_float64([1000])

    direct = _controller().solve(
        np.array([1.0, 0.0]), reference=np.zeros(2), previous_input=np.zeros(1)
    )
    assert control == pytest.approx(list(direct.proposed_input), abs=1.0e-9)
    slave.terminate()


def test_get_returns_inputs_and_reference() -> None:
    slave = _slave()
    slave.set_float64([0, 1, 2, 3], [0.7, -0.3, 0.1, 0.2])
    assert slave.get_float64([0, 1]) == [0.7, -0.3]
    assert slave.get_float64([2, 3]) == [0.1, 0.2]


def test_enter_initialization_resets_state() -> None:
    slave = _slave()
    slave.set_float64([0], [9.0])
    slave.enter_initialization_mode()
    assert slave.get_float64([0]) == [0.0]


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_set_rejects_a_length_mismatch() -> None:
    slave = _slave()
    with pytest.raises(ValueError, match="equal length"):
        slave.set_float64([0, 1], [1.0])


def test_set_rejects_an_output_reference() -> None:
    slave = _slave()
    with pytest.raises(ValueError, match="not an input"):
        slave.set_float64([1000], [1.0])


def test_set_rejects_an_unknown_reference() -> None:
    slave = _slave()
    with pytest.raises(ValueError, match="unknown value reference"):
        slave.set_float64([500], [1.0])


def test_do_step_rejects_a_negative_step() -> None:
    slave = _slave()
    with pytest.raises(ValueError, match="must be non-negative"):
        slave.do_step(0.0, -0.01)


# --------------------------------------------------------------------------- #
# modelDescription.xml FMI 3.0 conformance                                    #
# --------------------------------------------------------------------------- #
def test_model_description_is_conformant_fmi3() -> None:
    slave = _slave()
    document = generate_model_description(slave)
    assert document.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = fromstring(document)
    assert root.tag == "fmiModelDescription"
    assert root.get("fmiVersion") == "3.0"
    assert root.get("modelName") == "scpn_koopman_mpc"
    assert root.get("instantiationToken", "").startswith("{")
    cosim = root.find("CoSimulation")
    assert cosim is not None
    assert cosim.get("modelIdentifier") == "scpn_koopman_mpc"
    variables = root.find("ModelVariables")
    assert variables is not None
    assert all(child.tag == "Float64" for child in variables)
    assert len(variables) == 5
    structure = root.find("ModelStructure")
    assert structure is not None
    outputs = structure.findall("Output")
    assert len(outputs) == 1
    assert outputs[0].get("valueReference") == "1000"


def test_model_description_token_is_deterministic() -> None:
    first = generate_model_description(_slave())
    second = generate_model_description(_slave())
    assert first == second


# --------------------------------------------------------------------------- #
# .fmu packaging                                                              #
# --------------------------------------------------------------------------- #
def test_write_fmu_packages_the_model(tmp_path: Path) -> None:
    slave = _slave()
    destination = write_fmu(slave, tmp_path / "controller.fmu")
    assert destination.exists()
    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
        assert "modelDescription.xml" in names
        assert "resources/model.json" in names
        resources = json.loads(archive.read("resources/model.json"))
    assert resources["model_name"] == "scpn_koopman_mpc"
    assert resources["state_dim"] == 2
    assert resources["input_dim"] == 1
    # The serialised Koopman matrices reconstruct the predictor: A is the square
    # lifted transition, C maps the lift back to the two-dimensional state.
    lift_dim = len(resources["state_matrix"])
    assert all(len(row) == lift_dim for row in resources["state_matrix"])
    assert len(resources["output_matrix"]) == 2
