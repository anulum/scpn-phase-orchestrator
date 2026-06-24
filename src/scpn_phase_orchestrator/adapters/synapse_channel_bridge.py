# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SYNAPSE_CHANNEL WebSocket → SPO bridge

"""Live bridge from SYNAPSE_CHANNEL hub events to SPO phase dynamics.

Connects to the SYNAPSE_CHANNEL WebSocket hub and maps agent activity
into oscillator phases for real-time coherence monitoring.

Mapping:
- Heartbeat interval → P-channel frequency (regular = coherent)
- Task claim/release rate → I-channel frequency (balanced = coherent)
- Chat message similarity → S-channel coupling (same topic = coupled)

Usage::

    bridge = SynapseChannelBridge(
        hub_uri="ws://localhost:8876",
        agents=["Agent-A", "Agent-B", "Agent-C", "Human"],
    )
    await bridge.connect()

    # In your SPO loop:
    phases = bridge.get_phases()
    knm = bridge.get_coupling()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, TypeAlias, cast
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

__all__ = ["SynapseChannelBridge", "AgentState"]
logger = logging.getLogger(__name__)

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]
SynapseMessage: TypeAlias = dict[str, object]


def _reject_json_constant(value: str) -> None:
    """Raise if the JSON value is a forbidden constant."""
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_json_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Raise if the JSON object has duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        result[key] = value
    return result


def _loads_hub_json(payload: str) -> Any:
    """Load and validate a SYNAPSE hub JSON payload, else raise."""
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_key,
        )
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        raise ValueError(
            "Synapse hub JSON must contain finite JSON numbers and unique object keys"
        ) from exc


def _validate_hub_uri(hub_uri: str) -> str:
    """Validate Synapse hub websocket endpoint configuration."""
    if not isinstance(hub_uri, str) or not hub_uri:
        raise ValueError("hub_uri must be a non-empty ws(s) URL")

    parsed = urlparse(hub_uri)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("hub_uri must be a non-empty ws(s) URL")

    return hub_uri


def _validate_agents(agents: list[str] | None) -> list[str]:
    """Validate and copy the Synapse agent roster."""
    if agents is None:
        return []
    if not isinstance(agents, list):
        raise ValueError("agents must be a list of unique non-empty strings")

    seen: set[str] = set()
    validated: list[str] = []
    for agent in agents:
        if not isinstance(agent, str) or not agent:
            raise ValueError("agents must be a list of unique non-empty strings")
        if any(ord(char) < 32 for char in agent):
            raise ValueError("agent names must not contain control characters")
        if agent in seen:
            raise ValueError("agents must contain unique names")

        seen.add(agent)
        validated.append(agent)

    return validated


def _normalise_hub_message(value: object) -> SynapseMessage | None:
    """Return a safe hub message object or ``None`` for malformed payloads."""
    if not isinstance(value, dict):
        return None

    sender = value.get("sender")
    msg_type = value.get("type")
    if not isinstance(sender, str) or not sender:
        return None
    if not isinstance(msg_type, str) or not msg_type:
        return None
    if any(ord(char) < 32 for char in sender + msg_type):
        return None

    message: SynapseMessage = {"sender": sender, "type": msg_type}
    payload = value.get("payload")
    if msg_type == "claim_granted" and payload is None:
        return None
    if payload is not None:
        if not isinstance(payload, str) or not payload:
            return None
        if any(ord(char) < 32 for char in payload):
            return None
        message["payload"] = payload
    return message


@dataclass
class AgentState:
    """Tracked state for one agent."""

    last_heartbeat: float = 0.0
    heartbeat_intervals: list[float] = field(default_factory=list)
    task_events: list[float] = field(default_factory=list)
    message_count: int = 0
    current_task: str | None = None
    phase_p: float = 0.0  # P-channel: heartbeat rhythm
    phase_i: float = 0.0  # I-channel: task cadence
    phase_s: float = 0.0  # S-channel: topic alignment

    def __post_init__(self) -> None:
        if (
            not isinstance(self.last_heartbeat, Real)
            or isinstance(self.last_heartbeat, bool)
            or not isfinite(float(self.last_heartbeat))
            or float(self.last_heartbeat) < 0.0
        ):
            raise ValueError("last_heartbeat must be a finite non-negative real")
        self.last_heartbeat = float(self.last_heartbeat)

        if not isinstance(self.heartbeat_intervals, list):
            raise ValueError("heartbeat_intervals must be a list of finite positives")
        validated_intervals: list[float] = []
        for item in self.heartbeat_intervals:
            if (
                not isinstance(item, Real)
                or isinstance(item, bool)
                or not isfinite(float(item))
                or float(item) <= 0.0
            ):
                raise ValueError(
                    "heartbeat_intervals must be a list of finite positives"
                )
            validated_intervals.append(float(item))
        self.heartbeat_intervals = validated_intervals

        if not isinstance(self.task_events, list):
            raise ValueError("task_events must be a list of finite non-negative reals")
        validated_events: list[float] = []
        for item in self.task_events:
            if (
                not isinstance(item, Real)
                or isinstance(item, bool)
                or not isfinite(float(item))
                or float(item) < 0.0
            ):
                raise ValueError(
                    "task_events must be a list of finite non-negative reals"
                )
            validated_events.append(float(item))
        self.task_events = validated_events

        if (
            not isinstance(self.message_count, int)
            or isinstance(self.message_count, bool)
            or self.message_count < 0
        ):
            raise ValueError("message_count must be a non-negative integer")

        if self.current_task is not None:
            if not isinstance(self.current_task, str) or not self.current_task:
                raise ValueError("current_task must be None or a non-empty string")
            if any(ord(char) < 32 for char in self.current_task):
                raise ValueError("current_task must not contain control characters")

        for field_name in ("phase_p", "phase_i", "phase_s"):
            raw_value = getattr(self, field_name)
            if (
                not isinstance(raw_value, Real)
                or isinstance(raw_value, bool)
                or not isfinite(float(raw_value))
            ):
                raise ValueError(f"{field_name} must be finite")
            setattr(self, field_name, float(raw_value) % TWO_PI)


class SynapseChannelBridge:
    """Live bridge from SYNAPSE_CHANNEL to SPO oscillator phases."""

    def __init__(
        self,
        hub_uri: str = "ws://localhost:8876",
        agents: list[str] | None = None,
    ):
        self._uri = _validate_hub_uri(hub_uri)
        self._agents = _validate_agents(agents)
        self._agent_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._agents)
        }
        self._states: dict[str, AgentState] = {
            name: AgentState() for name in self._agents
        }
        self._ws: Any = None
        self._running = False
        self._n = len(self._agents)

    @property
    def n_oscillators(self) -> int:
        """Return the number of configured agent oscillators.

        Returns
        -------
        int
            Return the number of configured agent oscillators.
        """
        return self._n

    async def connect(self) -> None:
        """Connect to SYNAPSE_CHANNEL hub and start listening.

        Raises
        ------
        ImportError
            If the SYNAPSE channel client is not installed.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required: pip install websockets") from None

        self._ws = await websockets.connect(self._uri)
        # Register as observer
        await self._ws.send(
            json.dumps(
                {
                    "type": "chat",
                    "sender": "SPO-Bridge",
                    "target": "all",
                    "payload": "SPO coherence bridge connected",
                }
            )
        )
        self._running = True

    async def listen_once(self) -> None:
        """Process one message from the hub."""
        if self._ws is None:
            return
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
            msg = _loads_hub_json(raw)
            self._process_message(msg)
        except asyncio.TimeoutError:
            return  # no message within timeout — normal
        except (json.JSONDecodeError, ValueError):
            logger.warning("synapse.listen_once_invalid_json")
            return
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.warning(
                "synapse.listen_once_transport_error: %s", type(exc).__name__
            )
            return  # connection error — caller should retry

    def _process_message(self, msg: object) -> None:
        """Process an incoming SYNAPSE hub message."""
        message = _normalise_hub_message(msg)
        if message is None:
            return

        sender = cast(str, message["sender"])
        msg_type = cast(str, message["type"])
        now = time.time()

        if sender not in self._agent_idx:
            return

        state = self._states[sender]

        if msg_type == "heartbeat":
            if state.last_heartbeat > 0:
                interval = now - state.last_heartbeat
                state.heartbeat_intervals.append(interval)
                if len(state.heartbeat_intervals) > 20:
                    state.heartbeat_intervals.pop(0)
            state.last_heartbeat = now

        elif msg_type in ("claim_granted", "release_granted"):
            state.task_events.append(now)
            if len(state.task_events) > 20:
                state.task_events.pop(0)
            if msg_type == "claim_granted":
                state.current_task = str(message.get("payload", ""))
            else:
                state.current_task = None

        elif msg_type == "chat":
            state.message_count += 1

    def get_phases(self) -> FloatArray:
        """Extract current oscillator phases from agent activity.

        P-channel: phase advances at heartbeat frequency.
        I-channel: phase advances at task event frequency.
        S-channel: message count modulo 2π.

        Returns
        -------
        FloatArray
            Extract current oscillator phases from agent activity.
        """
        phases = np.zeros(self._n)
        time.time()

        for name, state in self._states.items():
            idx = self._agent_idx[name]

            # P: heartbeat regularity → phase
            if state.heartbeat_intervals:
                mean_interval = float(np.mean(state.heartbeat_intervals[-5:]))
                freq = 1.0 / max(mean_interval, 0.1)
                state.phase_p = (state.phase_p + TWO_PI * freq * 1.0) % TWO_PI
            phases[idx] = state.phase_p

        return phases

    def get_coupling(self) -> FloatArray:
        """Compute coupling from shared task context.

        Agents working on related tasks couple strongly.
        Agents with no task decouple.

        Returns
        -------
        FloatArray
            Compute coupling from shared task context.
        """
        knm = np.zeros((self._n, self._n))
        for name_i, state_i in self._states.items():
            for name_j, state_j in self._states.items():
                if name_i == name_j:
                    continue
                i = self._agent_idx[name_i]
                j = self._agent_idx[name_j]

                # Both active → couple
                if state_i.current_task and state_j.current_task:
                    knm[i, j] = 1.0
                # One idle → weak coupling
                elif state_i.current_task or state_j.current_task:
                    knm[i, j] = 0.3

        return knm

    def get_agent_summary(self) -> dict[str, dict[str, Any]]:
        """Return per-agent summary for display.

        Returns
        -------
        dict[str, dict[str, Any]]
            Return per-agent summary for display.
        """
        summary = {}
        for name, state in self._states.items():
            summary[name] = {
                "phase": state.phase_p,
                "task": state.current_task,
                "messages": state.message_count,
                "heartbeats": len(state.heartbeat_intervals),
            }
        return summary

    async def close(self) -> None:
        """Stop the bridge and close the active hub WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
