# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FMI 3.0 co-simulation export of the Koopman MPC

"""Export the Koopman MPC controller as an FMI 3.0 co-simulation slave.

The Functional Mock-up Interface (FMI 3.0, modelica.org) is the industrial
standard for coupling simulation tools. This adapter wraps the condensed Koopman
MPC (``actuation.koopman_mpc``) as an FMI co-simulation slave so a co-simulation
master — a power-systems or control bench such as Dymola, OpenModelica or FMPy —
can drive the SPO controller as a block: it sets the measured state and the set
point, calls ``do_step``, and reads back the proposed control.

The slave, the ``modelDescription.xml`` generator and the ``.fmu`` packager are
pure NumPy and fully exercised in-process by driving the slave the way a master
would. They emit a conformant FMI 3.0 model interface; loading the package inside
a third-party FMI tool additionally needs the C-ABI binary shim, which is an
optional, documented build step (the ``fmi`` extra) outside this module — the
review-only model and its evidence are produced here.

The reverse, import direction is :func:`cosimulate`: a co-simulation master that
drives the controller slave against a plant supplied as a step callable, closing
the loop. An external plant FMU plugs in by wrapping its FMI runtime (for example
``fmpy``) as that callable, so no FMI runtime dependency is imposed here either.

References
----------
* Modelica Association 2024, *Functional Mock-up Interface Specification 3.0*.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

# ElementTree is used only to BUILD the modelDescription document from trusted
# local model metadata, never to parse untrusted input, so the B405 parsing
# attack surface does not apply here.
from xml.etree import ElementTree  # nosec B405

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.koopman_mpc import KoopmanMPCController

FloatArray: TypeAlias = NDArray[np.float64]
PlantStep: TypeAlias = Callable[[FloatArray, FloatArray, float], FloatArray]

__all__ = [
    "CoSimulationSlave",
    "FMIVariable",
    "cosimulate",
    "generate_model_description",
    "write_fmu",
]

# Value-reference layout: state inputs occupy [0, n), reference inputs [n, 2n),
# control outputs start at this offset so the three blocks never collide.
_OUTPUT_VREF_BASE = 1000


@dataclass(frozen=True)
class FMIVariable:
    """A scalar FMI 3.0 ``Float64`` model variable.

    Parameters
    ----------
    name : str
        The variable name, a valid FMI identifier.
    value_reference : int
        The handle a co-simulation master uses to get or set the variable.
    causality : str
        ``"input"`` or ``"output"``.
    start : float | None
        The required start value for inputs; ``None`` for outputs.
    """

    name: str
    value_reference: int
    causality: str
    start: float | None = None


def _model_variables(state_dim: int, input_dim: int) -> tuple[FMIVariable, ...]:
    """Return the FMI model-variable definitions for the controller."""
    variables: list[FMIVariable] = []
    for index in range(state_dim):
        variables.append(FMIVariable(f"x_{index}", index, "input", 0.0))
    for index in range(state_dim):
        variables.append(FMIVariable(f"r_{index}", state_dim + index, "input", 0.0))
    for index in range(input_dim):
        variables.append(FMIVariable(f"u_{index}", _OUTPUT_VREF_BASE + index, "output"))
    return tuple(variables)


class CoSimulationSlave:
    """An FMI 3.0 co-simulation slave wrapping a Koopman MPC controller.

    The slave mirrors the FMI co-simulation lifecycle — set inputs, ``do_step``,
    get outputs — and computes the control by solving the MPC at each step.

    Parameters
    ----------
    controller : KoopmanMPCController
        The fitted Koopman MPC controller to expose.
    model_name : str
        The FMI model name.
    """

    def __init__(
        self, controller: KoopmanMPCController, *, model_name: str = "scpn_koopman_mpc"
    ) -> None:
        self._controller = controller
        self.model_name = model_name
        self.state_dim = int(controller.predictor.state_dim)
        self.input_dim = int(controller.predictor.input_dim)
        self.variables = _model_variables(self.state_dim, self.input_dim)
        self._state = np.zeros(self.state_dim, dtype=np.float64)
        self._reference = np.zeros(self.state_dim, dtype=np.float64)
        self._control = np.zeros(self.input_dim, dtype=np.float64)
        self._previous_input = np.zeros(self.input_dim, dtype=np.float64)
        self._by_reference = {var.value_reference: var for var in self.variables}

    def enter_initialization_mode(self) -> None:
        """Reset the slave to its start values for a fresh co-simulation run."""
        self._state = np.zeros(self.state_dim, dtype=np.float64)
        self._reference = np.zeros(self.state_dim, dtype=np.float64)
        self._control = np.zeros(self.input_dim, dtype=np.float64)
        self._previous_input = np.zeros(self.input_dim, dtype=np.float64)

    def exit_initialization_mode(self) -> None:
        """Compute the initial control output from the start inputs."""
        self._solve()

    def set_float64(self, value_references: list[int], values: list[float]) -> None:
        """Set input variables addressed by their value references.

        Parameters
        ----------
        value_references : list[int]
            The handles of the variables to set.
        values : list[float]
            The values, one per reference.

        Raises
        ------
        ValueError
            If the lengths differ, a reference is unknown, or it is not an input.
        """
        if len(value_references) != len(values):
            raise ValueError("value_references and values must have equal length")
        for reference, value in zip(value_references, values, strict=True):
            variable = self._lookup(reference)
            if variable.causality != "input":
                raise ValueError(f"value reference {reference} is not an input")
            if reference < self.state_dim:
                self._state[reference] = float(value)
            else:
                self._reference[reference - self.state_dim] = float(value)

    def get_float64(self, value_references: list[int]) -> list[float]:
        """Get any variables addressed by their value references.

        Parameters
        ----------
        value_references : list[int]
            The handles of the variables to read.

        Returns
        -------
        list[float]
            The current values, one per reference.

        Raises
        ------
        ValueError
            If a reference is unknown.
        """
        values: list[float] = []
        for reference in value_references:
            variable = self._lookup(reference)
            if variable.causality == "output":
                values.append(float(self._control[reference - _OUTPUT_VREF_BASE]))
            elif reference < self.state_dim:
                values.append(float(self._state[reference]))
            else:
                values.append(float(self._reference[reference - self.state_dim]))
        return values

    def do_step(
        self, current_communication_point: float, communication_step_size: float
    ) -> None:
        """Advance the co-simulation by solving the MPC for the current inputs.

        Parameters
        ----------
        current_communication_point : float
            The master's current time; recorded for the lifecycle contract.
        communication_step_size : float
            The communication step; the controller's own sample period governs
            the internal prediction, so the step is accepted as given.

        Raises
        ------
        ValueError
            If the communication step size is negative.
        """
        if communication_step_size < 0.0:
            raise ValueError("communication_step_size must be non-negative")
        self._solve()
        self._previous_input = self._control.copy()

    def terminate(self) -> None:
        """End the co-simulation; the slave holds no external resources."""

    def _solve(self) -> None:
        """Advance the co-simulation slave one communication step."""
        decision = self._controller.solve(
            self._state,
            reference=self._reference,
            previous_input=self._previous_input,
        )
        self._control = np.asarray(decision.proposed_input, dtype=np.float64)

    def _lookup(self, reference: int) -> FMIVariable:
        """Return the value reference for a named FMI variable, else raise."""
        variable = self._by_reference.get(reference)
        if variable is None:
            raise ValueError(f"unknown value reference {reference}")
        return variable


def _instantiation_token(model_name: str, variables: tuple[FMIVariable, ...]) -> str:
    """Return the FMI instantiation token for the slave."""
    payload = json.dumps(
        {
            "model": model_name,
            "variables": [
                [var.name, var.value_reference, var.causality] for var in variables
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (
        f"{{{digest[:8]}-{digest[8:12]}-{digest[12:16]}-"
        f"{digest[16:20]}-{digest[20:32]}}}"
    )


def generate_model_description(slave: CoSimulationSlave) -> str:
    """Render the FMI 3.0 ``modelDescription.xml`` for a slave.

    Parameters
    ----------
    slave : CoSimulationSlave
        The slave whose model interface to describe.

    Returns
    -------
    str
        The ``modelDescription.xml`` document.
    """
    root = ElementTree.Element(
        "fmiModelDescription",
        {
            "fmiVersion": "3.0",
            "modelName": slave.model_name,
            "instantiationToken": _instantiation_token(
                slave.model_name, slave.variables
            ),
            "description": "SCPN Koopman MPC controller (review-only proposal)",
            "generationTool": "SCPN Phase Orchestrator",
        },
    )
    ElementTree.SubElement(
        root,
        "CoSimulation",
        {
            "modelIdentifier": slave.model_name,
            "canHandleVariableCommunicationStepSize": "true",
        },
    )
    model_variables = ElementTree.SubElement(root, "ModelVariables")
    for variable in slave.variables:
        attributes = {
            "name": variable.name,
            "valueReference": str(variable.value_reference),
            "causality": variable.causality,
            "variability": "continuous",
        }
        if variable.start is not None:
            attributes["start"] = repr(variable.start)
        ElementTree.SubElement(model_variables, "Float64", attributes)
    model_structure = ElementTree.SubElement(root, "ModelStructure")
    for variable in slave.variables:
        if variable.causality == "output":
            reference = {"valueReference": str(variable.value_reference)}
            ElementTree.SubElement(model_structure, "Output", reference)
            ElementTree.SubElement(model_structure, "InitialUnknown", reference)
    ElementTree.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ElementTree.tostring(
        root, encoding="unicode"
    )


def write_fmu(slave: CoSimulationSlave, path: str | Path) -> Path:
    """Package a slave as a ``.fmu`` archive (model interface + resources).

    The archive carries the conformant ``modelDescription.xml`` and a
    ``resources/model.json`` describing the controller, the self-contained model
    a Python-backed FMI runtime reconstructs. Loading it inside a third-party FMI
    tool additionally needs the C-ABI binary shim from the optional ``fmi`` extra.

    Parameters
    ----------
    slave : CoSimulationSlave
        The slave to package.
    path : str | pathlib.Path
        Destination ``.fmu`` path.

    Returns
    -------
    pathlib.Path
        The written archive path.
    """
    destination = Path(path)
    resources = {
        "model_name": slave.model_name,
        "state_dim": slave.state_dim,
        "input_dim": slave.input_dim,
        "state_matrix": slave._controller.predictor.state_matrix.tolist(),
        "input_matrix": slave._controller.predictor.input_matrix.tolist(),
        "output_matrix": slave._controller.predictor.output_matrix.tolist(),
        "horizon": slave._controller.config.horizon,
    }
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("modelDescription.xml", generate_model_description(slave))
        archive.writestr("resources/model.json", json.dumps(resources, indent=2))
    return destination


def cosimulate(
    controller: CoSimulationSlave,
    plant_step: PlantStep,
    *,
    initial_state: FloatArray,
    steps: int,
    dt: float,
    reference: FloatArray | None = None,
) -> FloatArray:
    """Run a co-simulation master coupling the controller slave with a plant.

    This is the import/master direction: SPO drives a plant model in
    co-simulation. Each step writes the plant state to the controller's state
    inputs, advances the controller by one MPC step, reads its control output and
    applies it to the plant, then advances the plant. The plant is any step
    callable ``(state, control, dt) -> next_state``; an external plant FMU plugs
    in by wrapping its FMI runtime (for example ``fmpy``) as such a callable, so
    no FMI runtime dependency is imposed here.

    Parameters
    ----------
    controller : CoSimulationSlave
        The FMI controller slave to drive.
    plant_step : Callable[[numpy.ndarray, numpy.ndarray, float], numpy.ndarray]
        Advances the plant by ``dt`` under the applied control.
    initial_state : numpy.ndarray
        The plant's initial state ``x_0`` of shape ``(n,)``.
    steps : int
        Number of co-simulation steps.
    dt : float
        The communication step size.
    reference : numpy.ndarray | None
        The controller set point of shape ``(n,)``; defaults to the origin.

    Returns
    -------
    numpy.ndarray
        The closed-loop plant-state trajectory of shape ``(steps + 1, n)``.

    Raises
    ------
    ValueError
        If the initial state length, step count, or ``dt`` are inconsistent.
    """
    state_dim = controller.state_dim
    state = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64).ravel())
    if state.shape[0] != state_dim:
        raise ValueError("initial_state length must match the controller state")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    set_point = (
        np.zeros(state_dim, dtype=np.float64)
        if reference is None
        else np.asarray(reference, dtype=np.float64).ravel()
    )
    controller.enter_initialization_mode()
    controller.set_float64(
        list(range(state_dim, 2 * state_dim)), [float(v) for v in set_point]
    )
    controller.exit_initialization_mode()

    state_references = list(range(state_dim))
    control_references = [_OUTPUT_VREF_BASE + j for j in range(controller.input_dim)]
    trajectory = [state.copy()]
    time = 0.0
    for _ in range(steps):
        controller.set_float64(state_references, [float(v) for v in state])
        controller.do_step(time, dt)
        control = np.asarray(
            controller.get_float64(control_references), dtype=np.float64
        )
        state = np.ascontiguousarray(
            np.asarray(plant_step(state, control, dt), dtype=np.float64).ravel()
        )
        trajectory.append(state.copy())
        time += dt
    return np.asarray(trajectory, dtype=np.float64)
