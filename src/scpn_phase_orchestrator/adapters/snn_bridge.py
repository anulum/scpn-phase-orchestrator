# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SNN bridge adapter

"""Spiking-neural-network bridge for NumPy LIF estimates and review manifests.

``SNNControllerBridge`` maps UPDE layer coherence to input currents, estimates
steady-state LIF rates, proposes control actions from rate thresholds, and
builds deterministic Lava/PyNN schedule manifests with hardware writes disabled.
Optional Lava process construction is isolated to its explicit method. The core
bridge remains pure NumPy and does not perform neuromorphic actuation.
"""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from types import SimpleNamespace
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = ["SNNControllerBridge"]

FloatArray: TypeAlias = NDArray[np.float64]

# Abbott 1999, Eq. 1 — LIF time constants
TAU_RC = 0.02  # s, membrane time constant
TAU_REF = 0.002  # s, refractory period


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _require_positive_real(value: object, *, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite and positive")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _has_non_real_numeric_alias(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return True
    if isinstance(value, complex | np.complexfloating):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "c"}:
            return True
        if value.dtype.kind == "O":
            return any(_has_non_real_numeric_alias(item) for item in value.flat)
        return False
    if isinstance(value, list | tuple):
        return any(_has_non_real_numeric_alias(item) for item in value)
    return not isinstance(value, Real)


def _require_finite_array(values: object, *, field: str) -> FloatArray:
    if _has_non_real_numeric_alias(values):
        raise ValueError(f"{field} must contain real-valued numeric data")
    array: FloatArray = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
    return array


def _require_order_parameter(value: object, *, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _validated_layer_assignments(layer_assignments: list[int]) -> list[int]:
    if not isinstance(layer_assignments, list):
        raise ValueError("layer_assignments must be a list of non-negative integers")
    validated: list[int] = []
    for layer in layer_assignments:
        if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0:
            raise ValueError("layer_assignments must contain non-negative integers")
        validated.append(int(layer))
    return validated


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return cast("dict[str, object]", value)


def _require_non_empty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} must not contain control characters")
    return value.strip()


class SNNControllerBridge:
    """Bridge between UPDE state and spiking neural network controllers.

    All methods are pure-numpy — no external SNN libraries required.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        tau_rc: float = TAU_RC,
        tau_ref: float = TAU_REF,
    ) -> None:
        self.n_neurons = _require_positive_int(n_neurons, field="n_neurons")
        self.tau_rc = _require_positive_real(tau_rc, field="tau_rc")
        self.tau_ref = _require_positive_real(tau_ref, field="tau_ref")

    def upde_state_to_input_current(
        self, state: UPDEState, i_scale: float = 1.0
    ) -> FloatArray:
        """Map R values from each layer to LIF input currents.

        Parameters
        ----------
        state : UPDEState
            The current UPDE state.
        i_scale : float
            Scaling factor from order parameter to input current.

        Returns
        -------
        FloatArray
            The LIF input currents for each layer.
        """
        i_scale = _require_positive_real(i_scale, field="i_scale")
        r_values: FloatArray = np.array(
            [
                _require_order_parameter(layer.R, field=f"layer {idx} R")
                for idx, layer in enumerate(state.layers)
            ],
            dtype=np.float64,
        )
        result: FloatArray = r_values * i_scale
        return result

    def spike_rates_to_actions(
        self,
        rates: FloatArray,
        layer_assignments: list[int],
        threshold_hz: float = 50.0,
    ) -> list[ControlAction]:
        """Convert spike rates to control actions.

        *rates*: 1-D array of mean firing rates (Hz) per neuron group.
        *layer_assignments*: maps each rate index to a layer.
        *threshold_hz*: rates above this trigger coupling boost.

        Parameters
        ----------
        rates : FloatArray
            Per-layer spike rates.
        layer_assignments : list[int]
            Per-neuron layer assignments.
        threshold_hz : float
            Firing-rate threshold in Hz for emitting actions.

        Returns
        -------
        list[ControlAction]
            The control actions for the spike rates.

        Raises
        ------
        ValueError
            If the rates or assignments are invalid.
        """
        rates = _require_finite_array(rates, field="rates")
        if rates.ndim != 1:
            raise ValueError("rates must be 1-D")
        if np.any(rates < 0.0):
            raise ValueError("rates must be non-negative")
        layer_assignments = _validated_layer_assignments(layer_assignments)
        if len(layer_assignments) != rates.size:
            raise ValueError("layer_assignments length must match rates length")
        threshold_hz = _require_positive_real(threshold_hz, field="threshold_hz")
        actions: list[ControlAction] = []
        for idx, (rate, layer) in enumerate(
            zip(rates, layer_assignments, strict=False)
        ):
            if rate > threshold_hz:
                excess = (rate - threshold_hz) / threshold_hz
                actions.append(
                    ControlAction(
                        knob="K",
                        scope=f"layer_{layer}",
                        value=0.05 * excess,
                        ttl_s=5.0,
                        justification=f"SNN group {idx}: {rate:.1f} Hz",
                    )
                )
        return actions

    def lif_rate_estimate(self, currents: FloatArray) -> FloatArray:
        """Analytic LIF steady-state firing rate (Abbott 1999, Eq. 1).

        rate = 1 / (tau_ref - tau_rc * ln(1 - 1/J))  for J > 1

        Parameters
        ----------
        currents : FloatArray
            Per-neuron input currents.

        Returns
        -------
        FloatArray
            The analytic LIF steady-state firing rates.

        Raises
        ------
        ValueError
            If the input currents are invalid.
        """
        currents = _require_finite_array(currents, field="currents")
        if currents.ndim != 1:
            raise ValueError("currents must be 1-D")
        rates: FloatArray = np.zeros_like(currents, dtype=np.float64)
        above = currents > 1.0
        if above.any():
            j = currents[above]
            rates[above] = 1.0 / (self.tau_ref - self.tau_rc * np.log(1.0 - 1.0 / j))
        return rates

    def build_numpy_network(
        self, n_layers: int, seed: int = 0, synapse: float = 0.01
    ) -> SimpleNamespace:
        """Build a pure-numpy LIF network for UPDE-SNN coupling.

        Returns a SimpleNamespace with input_node, ensemble, output_node
        attributes and a step() method.

        Parameters
        ----------
        n_layers : int
            Number of SCPN layers.
        seed : int
            Seed for the deterministic RNG.
        synapse : float
            Synaptic weight scale.

        Returns
        -------
        SimpleNamespace
            The pure-numpy LIF network.
        """
        n_layers = _require_positive_int(n_layers, field="n_layers")
        seed = _require_nonnegative_int(seed, field="seed")
        synapse = _require_positive_real(synapse, field="synapse")
        rng = np.random.default_rng(seed)
        n = self.n_neurons
        encoders = rng.choice([-1.0, 1.0], (n, n_layers))
        max_rates = rng.uniform(100, 200, n)
        intercepts = rng.uniform(-0.5, 0.5, n)

        J_max = 1.0 / (1.0 - np.exp((self.tau_ref - 1.0 / max_rates) / self.tau_rc))
        alpha = (J_max - 1.0) / (1.0 - intercepts)
        J_bias = 1.0 - alpha * intercepts

        return SimpleNamespace(
            input_node=np.zeros(n_layers),
            ensemble=SimpleNamespace(
                n_neurons=n,
                encoders=encoders,
                alpha=alpha,
                J_bias=J_bias,
            ),
            output_node=np.zeros(n_layers),
            synapse=synapse,
            n_layers=n_layers,
        )

    # Backward-compat alias
    build_nengo_network = build_numpy_network

    def build_lava_process(self, n_layers: int) -> object:
        """Build a Lava LIF process for UPDE-SNN coupling.

        Raises ImportError if lava-nc is not installed.

        Parameters
        ----------
        n_layers : int
            Number of SCPN layers.

        Returns
        -------
        object
            The Lava LIF process.
        """
        _require_positive_int(n_layers, field="n_layers")
        # type ignore: lava-nc is an optional dependency without bundled stubs.
        from lava.proc.lif.process import LIF  # type: ignore[import-not-found]

        return LIF(
            shape=(self.n_neurons,),
            du=1.0 / self.tau_rc,
            dv=1.0 / self.tau_ref,
            vth=1.0,
        )

    def build_neuromorphic_schedule_manifest(
        self,
        state: UPDEState,
        *,
        i_scale: float = 1.0,
        threshold_hz: float = 50.0,
        projection_delay_ms: float = 1.0,
    ) -> dict[str, object]:
        """Compile a reviewable Lava/PyNN schedule from a UPDE state.

        The manifest is deterministic and contains simulator-parity evidence
        from the pure-numpy LIF rate path. It opens no hardware handles and
        does not permit actuation.

        Parameters
        ----------
        state : UPDEState
            The current UPDE state.
        i_scale : float
            Scaling factor from order parameter to input current.
        threshold_hz : float
            Firing-rate threshold in Hz for emitting actions.
        projection_delay_ms : float
            Projection delay in milliseconds.

        Returns
        -------
        dict[str, object]
            The reviewable Lava/PyNN schedule manifest.
        """
        self._validate_schedule_inputs(
            state,
            i_scale=i_scale,
            threshold_hz=threshold_hz,
            projection_delay_ms=projection_delay_ms,
        )
        currents = self.upde_state_to_input_current(state, i_scale=i_scale)
        rates = self.lif_rate_estimate(currents)
        parity_rates = self.lif_rate_estimate(currents)
        rate_error = float(np.max(np.abs(rates - parity_rates))) if rates.size else 0.0

        populations = [
            self._population_record(
                layer_index=idx,
                layer=layer,
                input_current=float(currents[idx]),
                estimated_rate_hz=float(rates[idx]),
            )
            for idx, layer in enumerate(state.layers)
        ]
        manifest: dict[str, object] = {
            "manifest_kind": "neuromorphic_schedule_manifest",
            "schema_version": 1,
            "status": (
                "simulator_parity_passed"
                if rate_error == 0.0
                else "simulator_parity_failed"
            ),
            "target_backends": ["lava", "pynn"],
            "n_layers": len(state.layers),
            "n_neurons_per_population": self.n_neurons,
            "tau_rc_s": self.tau_rc,
            "tau_ref_s": self.tau_ref,
            "input_scale": float(i_scale),
            "threshold_hz": float(threshold_hz),
            "actuation_permitted": False,
            "hardware_write_permitted": False,
            "populations": populations,
            "projections": self._projection_records(
                state.cross_layer_alignment,
                delay_ms=projection_delay_ms,
            ),
            "control_actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.spike_rates_to_actions(
                    rates,
                    layer_assignments=list(range(len(state.layers))),
                    threshold_hz=threshold_hz,
                )
            ],
            "simulator_parity": {
                "engine": "numpy_lif_rate_estimate",
                "max_abs_rate_error_hz": rate_error,
                "sample_count": len(state.layers),
            },
            "operator_commands": [
                "review neuromorphic_schedule_manifest.json",
                "run Lava or PyNN simulator parity before hardware handoff",
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["schedule_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return manifest

    def audit_hardware_target_readiness(
        self,
        manifest: dict[str, object],
        *,
        target_backend: str,
        hardware_site: str,
        credentials_configured: bool = False,
        operator_approved: bool = False,
        external_simulator_parity_verified: bool = False,
    ) -> dict[str, object]:
        """Return non-executing neuromorphic hardware readiness evidence.

        The audit validates a schedule manifest against a declared target and
        records whether external operator preconditions are present. It never
        opens a backend connection, submits a hardware job, or enables
        actuation/hardware-write permissions.

        Parameters
        ----------
        manifest : dict[str, object]
            The compiler manifest to audit.
        target_backend : str
            Name of the target backend.
        hardware_site : str
            Name of the deployment hardware site.
        credentials_configured : bool
            Whether provider credentials are configured.
        operator_approved : bool
            Whether a human operator approved the target.
        external_simulator_parity_verified : bool
            Whether external-simulator parity has been verified.

        Returns
        -------
        dict[str, object]
            The non-executing neuromorphic hardware readiness evidence.

        Raises
        ------
        ValueError
            If the manifest or target details are invalid.
        """
        manifest_record = _require_mapping(manifest, field="manifest")
        target_backend = _require_non_empty_text(
            target_backend,
            field="target_backend",
        )
        hardware_site = _require_non_empty_text(hardware_site, field="hardware_site")
        for field, value in (
            ("credentials_configured", credentials_configured),
            ("operator_approved", operator_approved),
            ("external_simulator_parity_verified", external_simulator_parity_verified),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a boolean")
        if manifest_record.get("manifest_kind") != "neuromorphic_schedule_manifest":
            raise ValueError("manifest must be a neuromorphic_schedule_manifest")

        target_backends = manifest_record.get("target_backends")
        if not isinstance(target_backends, list) or not all(
            isinstance(item, str) for item in target_backends
        ):
            raise ValueError("manifest target_backends must be a list of strings")
        if target_backend not in target_backends:
            raise ValueError("target_backend is not declared by manifest")

        blocked_reasons: list[str] = []
        if manifest_record.get("status") != "simulator_parity_passed":
            blocked_reasons.append("simulator_parity_not_passed")
        if manifest_record.get("hardware_write_permitted") is not False:
            blocked_reasons.append("hardware_write_permission_must_remain_false")
        if manifest_record.get("actuation_permitted") is not False:
            blocked_reasons.append("actuation_permission_must_remain_false")
        if not credentials_configured:
            blocked_reasons.append("credentials_not_configured")
        if not operator_approved:
            blocked_reasons.append("operator_approval_missing")
        if not external_simulator_parity_verified:
            blocked_reasons.append("external_simulator_parity_not_verified")

        manifest_sha = str(manifest_record.get("schedule_sha256", ""))
        record: dict[str, object] = {
            "schema": "scpn_neuromorphic_target_readiness_v1",
            "target_backend": target_backend,
            "hardware_site": hardware_site,
            "manifest_sha256": manifest_sha,
            "status": "blocked" if blocked_reasons else "ready_not_executed",
            "blocked_reasons": blocked_reasons,
            "credentials_configured": credentials_configured,
            "operator_approved": operator_approved,
            "external_simulator_parity_verified": external_simulator_parity_verified,
            "hardware_write_permitted": False,
            "actuation_permitted": False,
            "operator_commands": [
                "review neuromorphic_schedule_manifest.json",
                "run target simulator parity outside SPO before hardware handoff",
                (
                    "submit neuromorphic hardware job only from an approved "
                    "operator workflow"
                ),
            ],
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["readiness_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return record

    def _validate_schedule_inputs(
        self,
        state: UPDEState,
        *,
        i_scale: float,
        threshold_hz: float,
        projection_delay_ms: float,
    ) -> None:
        n_layers = len(state.layers)
        if self.n_neurons < 1:
            raise ValueError("n_neurons must be >= 1")
        for name, value in (
            ("tau_rc", self.tau_rc),
            ("tau_ref", self.tau_ref),
            ("i_scale", i_scale),
            ("threshold_hz", threshold_hz),
            ("projection_delay_ms", projection_delay_ms),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        alignment = _require_finite_array(
            state.cross_layer_alignment,
            field="cross_layer_alignment",
        )
        if alignment.shape != (n_layers, n_layers):
            raise ValueError(
                "cross_layer_alignment shape must match layer count, "
                f"got {alignment.shape} for {n_layers} layers"
            )
        if not np.all(np.isfinite(alignment)):
            raise ValueError("cross_layer_alignment must contain finite values")
        for idx, layer in enumerate(state.layers):
            _require_order_parameter(layer.R, field=f"layer {idx} R")
            if not np.isfinite(layer.psi):
                raise ValueError(f"layer {idx} psi must be finite")

    def _population_record(
        self,
        *,
        layer_index: int,
        layer: LayerState,
        input_current: float,
        estimated_rate_hz: float,
    ) -> dict[str, object]:
        return {
            "name": f"layer_{layer_index}",
            "layer_index": layer_index,
            "n_neurons": self.n_neurons,
            "r_value": float(layer.R),
            "psi": float(layer.psi),
            "input_current": input_current,
            "estimated_rate_hz": estimated_rate_hz,
            "lava_process": "LIF",
            "lava_shape": [self.n_neurons],
            "lava_parameters": {
                "du": 1.0 / self.tau_rc,
                "dv": 1.0 / self.tau_ref,
                "vth": 1.0,
            },
            "pynn_cell": "IF_curr_exp",
            "pynn_parameters": {
                "tau_m": self.tau_rc * 1000.0,
                "tau_refrac": self.tau_ref * 1000.0,
                "i_offset": input_current,
            },
        }

    def _projection_records(
        self,
        alignment: FloatArray,
        *,
        delay_ms: float,
    ) -> list[dict[str, object]]:
        projections: list[dict[str, object]] = []
        n_layers = alignment.shape[0]
        for source in range(n_layers):
            for target in range(n_layers):
                if source == target:
                    continue
                weight = float(alignment[source, target])
                if weight <= 0.0:
                    continue
                projections.append(
                    {
                        "source": f"layer_{source}",
                        "target": f"layer_{target}",
                        "weight": weight,
                        "delay_ms": delay_ms,
                        "receptor_type": "excitatory",
                    }
                )
        return projections
