# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OPC-UA SCADA tag bridge

"""OPC-UA bridge for industrial SCADA/DCS phase extraction.

Reads oscillator-relevant process tags (temperatures, pressures, flow rates)
from an OPC-UA server and maps each tag's sampled waveform to a physical-channel
phase state via the Hilbert transform (:class:`PhysicalExtractor`).

The bridge separates three concerns so the bulk is testable without a server:

* **Configuration** — :class:`OpcUaTag` and :class:`OpcUaBridgeConfig` validate
  the endpoint URL, tag declarations, and security posture eagerly.
* **Phase extraction** — :meth:`OpcUaPhaseBridge.extract_phases` turns decoded
  tag sample series into per-tag :class:`PhaseState` objects with no network or
  ``asyncua`` dependency, and :meth:`OpcUaPhaseBridge.collect_samples` polls an
  injected synchronous reader callable.
* **Live read** — :meth:`OpcUaPhaseBridge.read_live` and
  :meth:`OpcUaPhaseBridge.collect_live` use ``asyncua`` (optional dependency,
  ``opcua`` extra) to read node values from a connected client.

The bridge never writes to the OPC-UA server; it is a read-only ingestion path.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_phase_orchestrator.adapters._schema import require_non_empty_str
from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

if TYPE_CHECKING:
    from asyncua import Client

__all__ = [
    "HAS_ASYNCUA",
    "OpcUaBridgeConfig",
    "OpcUaPhaseBridge",
    "OpcUaTag",
]

# ``asyncua`` (and its ``cryptography`` dependency) is imported lazily inside
# ``connect`` so that merely importing this module — e.g. under coverage — does
# not pull a heavy optional dependency that can perturb NumPy/SciPy C-extension
# loading. Availability is detected without importing the package.
HAS_ASYNCUA = importlib.util.find_spec("asyncua") is not None

_VALID_CHANNELS = frozenset({"P", "R", "E", "S"})
_VALID_SECURITY_POLICIES = frozenset(
    {"None", "Basic256Sha256", "Aes128Sha256RsaOaep", "Aes256Sha256RsaPss"}
)
_VALID_SECURITY_MODES = frozenset({"None", "Sign", "SignAndEncrypt"})


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _positive_real(value: object, *, field_name: str) -> float:
    number = _finite_real(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return number


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class OpcUaTag:
    """Declares how one OPC-UA node maps to a physical oscillator.

    Attributes
    ----------
    node_id : str
        OPC-UA node identifier (e.g. ``"ns=2;i=4"`` or ``"ns=2;s=Reactor.Temp"``).
    name : str
        Oscillator name the extracted phase state is bound to.
    channel : str
        SPO channel label; one of ``"P"``, ``"R"``, ``"E"``, ``"S"`` (default
        ``"P"`` for the physical channel).
    scale, offset : float
        Affine calibration applied to each raw sample as ``scale * x + offset``.
    sample_rate_hz : float
        Sampling rate of the tag waveform in hertz, used by the Hilbert phase
        extraction.
    """

    node_id: str
    name: str
    channel: str = "P"
    scale: float = 1.0
    offset: float = 0.0
    sample_rate_hz: float = 1.0

    def __post_init__(self) -> None:
        require_non_empty_str(self.node_id, field="node_id")
        require_non_empty_str(self.name, field="name")
        if self.channel not in _VALID_CHANNELS:
            raise ValueError(f"channel must be one of {sorted(_VALID_CHANNELS)}")
        _finite_real(self.scale, field_name="scale")
        _finite_real(self.offset, field_name="offset")
        _positive_real(self.sample_rate_hz, field_name="sample_rate_hz")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the tag.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the tag fields.
        """
        return {
            "node_id": self.node_id,
            "name": self.name,
            "channel": self.channel,
            "scale": self.scale,
            "offset": self.offset,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass(frozen=True)
class OpcUaBridgeConfig:
    """Validated OPC-UA connection and tag-mapping configuration.

    Attributes
    ----------
    endpoint_url : str
        OPC-UA endpoint, must use the ``opc.tcp://`` scheme.
    tags : tuple[OpcUaTag, ...]
        The tags to read; node identifiers and oscillator names must be unique.
    security_policy : str
        OPC-UA security policy (default ``"None"``).
    security_mode : str
        Message security mode (default ``"None"``).
    request_timeout_s : float
        Per-request timeout in seconds for the live client.
    """

    endpoint_url: str
    tags: tuple[OpcUaTag, ...]
    security_policy: str = "None"
    security_mode: str = "None"
    request_timeout_s: float = 4.0

    def __post_init__(self) -> None:
        require_non_empty_str(self.endpoint_url, field="endpoint_url")
        if not self.endpoint_url.startswith("opc.tcp://"):
            raise ValueError("endpoint_url must use the opc.tcp:// scheme")
        if not self.tags:
            raise ValueError("at least one tag must be declared")
        if self.security_policy not in _VALID_SECURITY_POLICIES:
            raise ValueError(
                f"security_policy must be one of {sorted(_VALID_SECURITY_POLICIES)}"
            )
        if self.security_mode not in _VALID_SECURITY_MODES:
            raise ValueError(
                f"security_mode must be one of {sorted(_VALID_SECURITY_MODES)}"
            )
        _positive_real(self.request_timeout_s, field_name="request_timeout_s")
        node_ids = [tag.node_id for tag in self.tags]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("tag node_ids must be unique")
        names = [tag.name for tag in self.tags]
        if len(set(names)) != len(names):
            raise ValueError("tag names must be unique")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the configuration.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the configuration fields.
        """
        return {
            "endpoint_url": self.endpoint_url,
            "tags": [tag.to_audit_record() for tag in self.tags],
            "security_policy": self.security_policy,
            "security_mode": self.security_mode,
            "request_timeout_s": self.request_timeout_s,
        }


@dataclass
class OpcUaPhaseBridge:
    """Read-only OPC-UA tag ingestion mapped to physical phase states.

    Attributes
    ----------
    config : OpcUaBridgeConfig
        The validated bridge configuration.
    """

    config: OpcUaBridgeConfig
    _extractors: dict[str, PhysicalExtractor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._extractors = {
            tag.name: PhysicalExtractor(node_id=tag.name) for tag in self.config.tags
        }

    @classmethod
    def from_tags(
        cls,
        endpoint_url: str,
        tags: Sequence[OpcUaTag],
        **config_kwargs: object,
    ) -> OpcUaPhaseBridge:
        """Build a bridge from an endpoint and a tag sequence.

        Parameters
        ----------
        endpoint_url : str
            OPC-UA endpoint (``opc.tcp://`` scheme).
        tags : Sequence[OpcUaTag]
            The tags to read.
        **config_kwargs : object
            Forwarded to :class:`OpcUaBridgeConfig` (security and timeout).

        Returns
        -------
        OpcUaPhaseBridge
            A configured bridge.
        """
        config = OpcUaBridgeConfig(
            endpoint_url=endpoint_url,
            tags=tuple(tags),
            **config_kwargs,  # type: ignore[arg-type]
        )
        return cls(config=config)

    def extract_phases(
        self,
        tag_samples: Mapping[str, Sequence[float]],
    ) -> dict[str, PhaseState]:
        """Map decoded tag sample series to per-tag physical phase states.

        Parameters
        ----------
        tag_samples : Mapping[str, Sequence[float]]
            Sample series keyed by tag name; every declared tag must be present
            with at least one finite sample.

        Returns
        -------
        dict[str, PhaseState]
            The latest instantaneous phase state per tag, keyed by tag name.

        Raises
        ------
        ValueError
            If a declared tag is missing, a series is empty, or a sample is not a
            finite real value.
        """
        phases: dict[str, PhaseState] = {}
        for tag in self.config.tags:
            if tag.name not in tag_samples:
                raise ValueError(f"missing samples for tag {tag.name!r}")
            raw = tag_samples[tag.name]
            if len(raw) == 0:
                raise ValueError(f"tag {tag.name!r} has no samples")
            calibrated = np.asarray(
                [_finite_real(value, field_name=f"{tag.name} sample") for value in raw],
                dtype=np.float64,
            )
            calibrated = tag.scale * calibrated + tag.offset
            states = self._extractors[tag.name].extract(calibrated, tag.sample_rate_hz)
            phases[tag.name] = states[0]
        return phases

    def collect_samples(
        self,
        reader: Callable[[str], float],
        *,
        samples_per_tag: int,
    ) -> dict[str, list[float]]:
        """Poll a synchronous reader callable into per-tag sample series.

        Parameters
        ----------
        reader : Callable[[str], float]
            Returns the current value for a node identifier; called
            ``samples_per_tag`` times per tag in declaration order.
        samples_per_tag : int
            Number of samples to collect per tag.

        Returns
        -------
        dict[str, list[float]]
            Sample series keyed by tag name.

        Raises
        ------
        ValueError
            If ``samples_per_tag`` is not a positive integer or a read value is
            not a finite real number.
        """
        count = _positive_int(samples_per_tag, field_name="samples_per_tag")
        samples: dict[str, list[float]] = {tag.name: [] for tag in self.config.tags}
        for _ in range(count):
            for tag in self.config.tags:
                value = _finite_real(
                    reader(tag.node_id), field_name=f"{tag.name} reading"
                )
                samples[tag.name].append(value)
        return samples

    def connect(self) -> Client:
        """Create a (not yet connected) ``asyncua`` client for the endpoint.

        Returns
        -------
        asyncua.Client
            A client configured for the endpoint and request timeout. Use it as
            an async context manager to open and close the connection.

        Raises
        ------
        RuntimeError
            If the optional ``asyncua`` dependency is not installed.
        """
        if not HAS_ASYNCUA:
            raise RuntimeError(
                "asyncua is not installed; install the 'opcua' extra to read live"
            )
        from asyncua import Client as AsyncuaClient

        return AsyncuaClient(
            url=self.config.endpoint_url,
            timeout=self.config.request_timeout_s,
        )

    async def read_live(
        self,
        client: Any,
        *,
        samples_per_tag: int,
        interval_s: float = 0.0,
    ) -> dict[str, list[float]]:
        """Read node values from a connected ``asyncua`` client.

        Parameters
        ----------
        client : asyncua.Client
            A connected client (or any object exposing ``get_node(node_id)``
            with an awaitable ``read_value``).
        samples_per_tag : int
            Number of samples to read per tag.
        interval_s : float, optional
            Delay in seconds between sampling rounds (default ``0``).

        Returns
        -------
        dict[str, list[float]]
            Sample series keyed by tag name.

        Raises
        ------
        ValueError
            If ``samples_per_tag`` is not positive, ``interval_s`` is negative, or
            a read value is not a finite real number.
        """
        count = _positive_int(samples_per_tag, field_name="samples_per_tag")
        interval = _finite_real(interval_s, field_name="interval_s")
        if interval < 0.0:
            raise ValueError("interval_s must be >= 0")
        nodes = {tag.name: client.get_node(tag.node_id) for tag in self.config.tags}
        samples: dict[str, list[float]] = {tag.name: [] for tag in self.config.tags}
        for index in range(count):
            if index > 0 and interval > 0.0:
                await asyncio.sleep(interval)
            for tag in self.config.tags:
                value = await nodes[tag.name].read_value()
                samples[tag.name].append(
                    _finite_real(value, field_name=f"{tag.name} reading")
                )
        return samples

    async def collect_live(
        self,
        *,
        samples_per_tag: int,
        interval_s: float = 0.0,
    ) -> dict[str, list[float]]:
        """Connect, read ``samples_per_tag`` samples per tag, and disconnect.

        Parameters
        ----------
        samples_per_tag : int
            Number of samples to read per tag.
        interval_s : float, optional
            Delay in seconds between sampling rounds (default ``0``).

        Returns
        -------
        dict[str, list[float]]
            Sample series keyed by tag name.

        Raises
        ------
        RuntimeError
            If the optional ``asyncua`` dependency is not installed.
        """
        client = self.connect()
        async with client:
            return await self.read_live(
                client,
                samples_per_tag=samples_per_tag,
                interval_s=interval_s,
            )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the bridge configuration.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping with the configuration and the
            ``asyncua`` availability flag.
        """
        return {
            "config": self.config.to_audit_record(),
            "asyncua_available": HAS_ASYNCUA,
        }
