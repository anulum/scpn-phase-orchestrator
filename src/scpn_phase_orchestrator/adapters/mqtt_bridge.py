# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MQTT edge sensor bridge

"""MQTT bridge for edge/IoT sensor phase extraction.

Subscribes to MQTT topics carrying oscillator-relevant process measurements and
maps each topic's sampled waveform to a physical-channel phase state via the
Hilbert transform (:class:`PhysicalExtractor`).

Like the OPC-UA bridge, the bulk is testable without a broker:

* **Configuration** — :class:`MqttTag` and :class:`MqttBridgeConfig` validate the
  broker endpoint and topic mapping eagerly.
* **Decoding and ingestion** — :meth:`MqttPhaseBridge.decode_payload` parses raw
  or JSON payloads, :meth:`MqttPhaseBridge.ingest_messages` folds a batch of
  ``(topic, payload)`` messages into per-tag sample series, and
  :meth:`MqttPhaseBridge.extract_phases` turns those series into
  :class:`PhaseState` objects — all with no network or ``paho-mqtt`` dependency.
* **Live subscribe** — :meth:`MqttPhaseBridge.collect_live` uses ``paho-mqtt``
  (optional dependency, ``mqtt`` extra) to subscribe and accumulate messages.

The bridge is read-only: it never publishes to the broker.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_tcp_port,
)
from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

if TYPE_CHECKING:
    from paho.mqtt.client import Client

__all__ = [
    "HAS_PAHO_MQTT",
    "MqttBridgeConfig",
    "MqttPhaseBridge",
    "MqttTag",
]

# ``paho-mqtt`` is imported lazily inside ``connect`` so importing this module —
# for example under coverage — does not pull the optional dependency.
HAS_PAHO_MQTT = importlib.util.find_spec("paho.mqtt.client") is not None

_VALID_CHANNELS = frozenset({"P", "R", "E", "S"})
_VALID_PAYLOAD_FORMATS = frozenset({"raw", "json"})


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
class MqttTag:
    """Declares how one MQTT topic maps to a physical oscillator.

    Attributes
    ----------
    topic : str
        MQTT topic the sensor publishes to.
    name : str
        Oscillator name the extracted phase state is bound to.
    channel : str
        SPO channel label; one of ``"P"``, ``"R"``, ``"E"``, ``"S"``.
    scale, offset : float
        Affine calibration applied to each decoded value as ``scale * x + offset``.
    sample_rate_hz : float
        Publish rate of the topic in hertz, used by the Hilbert phase extraction.
    payload_format : str
        ``"raw"`` (a decimal number as text) or ``"json"`` (a JSON number or a
        JSON object with a ``"value"`` field).
    """

    topic: str
    name: str
    channel: str = "P"
    scale: float = 1.0
    offset: float = 0.0
    sample_rate_hz: float = 1.0
    payload_format: str = "raw"

    def __post_init__(self) -> None:
        require_non_empty_str(self.topic, field="topic")
        require_non_empty_str(self.name, field="name")
        if self.channel not in _VALID_CHANNELS:
            raise ValueError(f"channel must be one of {sorted(_VALID_CHANNELS)}")
        _finite_real(self.scale, field_name="scale")
        _finite_real(self.offset, field_name="offset")
        _positive_real(self.sample_rate_hz, field_name="sample_rate_hz")
        if self.payload_format not in _VALID_PAYLOAD_FORMATS:
            raise ValueError(
                f"payload_format must be one of {sorted(_VALID_PAYLOAD_FORMATS)}"
            )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the tag.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the tag fields.
        """
        return {
            "topic": self.topic,
            "name": self.name,
            "channel": self.channel,
            "scale": self.scale,
            "offset": self.offset,
            "sample_rate_hz": self.sample_rate_hz,
            "payload_format": self.payload_format,
        }


@dataclass(frozen=True)
class MqttBridgeConfig:
    """Validated MQTT connection and topic-mapping configuration.

    Attributes
    ----------
    broker_host : str
        MQTT broker hostname or address.
    tags : tuple[MqttTag, ...]
        Topics to subscribe to; topics and oscillator names must be unique.
    broker_port : int
        Broker TCP port (default ``1883``).
    keepalive_s : int
        Keep-alive interval in seconds for the live client.
    client_id : str
        MQTT client identifier.
    use_tls : bool
        Whether the live client should negotiate TLS.
    """

    broker_host: str
    tags: tuple[MqttTag, ...]
    broker_port: int = 1883
    keepalive_s: int = 60
    client_id: str = "spo-mqtt-bridge"
    use_tls: bool = False

    def __post_init__(self) -> None:
        require_non_empty_str(self.broker_host, field="broker_host")
        require_tcp_port(self.broker_port, field="broker_port")
        _positive_int(self.keepalive_s, field_name="keepalive_s")
        require_non_empty_str(self.client_id, field="client_id")
        if not self.tags:
            raise ValueError("at least one tag must be declared")
        topics = [tag.topic for tag in self.tags]
        if len(set(topics)) != len(topics):
            raise ValueError("tag topics must be unique")
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
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "keepalive_s": self.keepalive_s,
            "client_id": self.client_id,
            "use_tls": self.use_tls,
            "tags": [tag.to_audit_record() for tag in self.tags],
        }


@dataclass
class MqttPhaseBridge:
    """Read-only MQTT topic ingestion mapped to physical phase states.

    Attributes
    ----------
    config : MqttBridgeConfig
        The validated bridge configuration.
    """

    config: MqttBridgeConfig
    _extractors: dict[str, PhysicalExtractor] = field(init=False, repr=False)
    _by_topic: dict[str, MqttTag] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._extractors = {
            tag.name: PhysicalExtractor(node_id=tag.name) for tag in self.config.tags
        }
        self._by_topic = {tag.topic: tag for tag in self.config.tags}

    @classmethod
    def from_tags(
        cls,
        broker_host: str,
        tags: Sequence[MqttTag],
        **config_kwargs: object,
    ) -> MqttPhaseBridge:
        """Build a bridge from a broker host and a tag sequence.

        Parameters
        ----------
        broker_host : str
            MQTT broker hostname or address.
        tags : Sequence[MqttTag]
            The topics to subscribe to.
        **config_kwargs : object
            Forwarded to :class:`MqttBridgeConfig`.

        Returns
        -------
        MqttPhaseBridge
            A configured bridge.
        """
        config = MqttBridgeConfig(
            broker_host=broker_host,
            tags=tuple(tags),
            # type ignore: forwarded **kwargs are validated by
            # MqttBridgeConfig.__post_init__ at construction time.
            **config_kwargs,  # type: ignore[arg-type]
        )
        return cls(config=config)

    def decode_payload(self, tag: MqttTag, payload: bytes | str) -> float:
        """Decode one MQTT payload into a finite calibrated sample value.

        Parameters
        ----------
        tag : MqttTag
            The tag whose ``payload_format`` and calibration apply.
        payload : bytes or str
            The raw message payload.

        Returns
        -------
        float
            The decoded ``scale * value + offset`` sample.

        Raises
        ------
        ValueError
            If the payload cannot be decoded to a finite real value.
        """
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        if tag.payload_format == "json":
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tag {tag.name!r} payload is not valid JSON") from exc
            if isinstance(decoded, Mapping):
                decoded = decoded.get("value")
            raw = _finite_real(decoded, field_name=f"{tag.name} value")
        else:
            try:
                raw = float(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tag {tag.name!r} payload is not a number: {text!r}"
                ) from exc
            if not np.isfinite(raw):
                raise ValueError(f"tag {tag.name!r} payload must be finite")
        return tag.scale * raw + tag.offset

    def ingest_messages(
        self,
        messages: Sequence[tuple[str, bytes | str]],
    ) -> dict[str, list[float]]:
        """Fold a batch of ``(topic, payload)`` messages into per-tag series.

        Messages on unknown topics are ignored. Decoded values are appended in
        arrival order to the series of the tag matching their topic.

        Parameters
        ----------
        messages : Sequence[tuple[str, bytes | str]]
            The received messages.

        Returns
        -------
        dict[str, list[float]]
            Calibrated sample series keyed by tag name (empty list per tag with
            no matching messages).
        """
        samples: dict[str, list[float]] = {tag.name: [] for tag in self.config.tags}
        for topic, payload in messages:
            tag = self._by_topic.get(topic)
            if tag is None:
                continue
            samples[tag.name].append(self.decode_payload(tag, payload))
        return samples

    def extract_phases(
        self,
        topic_samples: Mapping[str, Sequence[float]],
    ) -> dict[str, PhaseState]:
        """Map per-tag sample series to physical phase states.

        Parameters
        ----------
        topic_samples : Mapping[str, Sequence[float]]
            Calibrated sample series keyed by tag name; every declared tag must be
            present with at least one finite sample.

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
            if tag.name not in topic_samples:
                raise ValueError(f"missing samples for tag {tag.name!r}")
            raw = topic_samples[tag.name]
            if len(raw) == 0:
                raise ValueError(f"tag {tag.name!r} has no samples")
            series = np.asarray(
                [_finite_real(value, field_name=f"{tag.name} sample") for value in raw],
                dtype=np.float64,
            )
            states = self._extractors[tag.name].extract(series, tag.sample_rate_hz)
            phases[tag.name] = states[0]
        return phases

    def connect(self) -> Client:
        """Create a configured (not yet connected) ``paho-mqtt`` client.

        Returns
        -------
        paho.mqtt.client.Client
            A client bound to the configured client id and TLS posture.

        Raises
        ------
        RuntimeError
            If the optional ``paho-mqtt`` dependency is not installed.
        """
        if not HAS_PAHO_MQTT:
            raise RuntimeError(
                "paho-mqtt is not installed; install the 'mqtt' extra to read live"
            )
        from paho.mqtt.client import CallbackAPIVersion, Client

        client = Client(CallbackAPIVersion.VERSION2, client_id=self.config.client_id)
        if self.config.use_tls:
            client.tls_set()
        return client

    def collect_live(
        self,
        *,
        samples_per_tag: int,
        timeout_s: float = 10.0,
        client: Any = None,
    ) -> dict[str, list[float]]:
        """Subscribe to the configured topics and accumulate samples.

        Parameters
        ----------
        samples_per_tag : int
            Target number of samples to collect for every tag before returning.
        timeout_s : float, optional
            Maximum seconds to wait for the target to be reached (default ``10``).
        client : paho.mqtt.client.Client, optional
            An existing client; a new one is created via :meth:`connect` when
            omitted.

        Returns
        -------
        dict[str, list[float]]
            Sample series keyed by tag name (each truncated to at most
            ``samples_per_tag``).

        Raises
        ------
        ValueError
            If ``samples_per_tag`` is not positive or ``timeout_s`` is not finite
            and positive.
        """
        count = _positive_int(samples_per_tag, field_name="samples_per_tag")
        deadline = _positive_real(timeout_s, field_name="timeout_s")
        active = client if client is not None else self.connect()
        samples: dict[str, list[float]] = {tag.name: [] for tag in self.config.tags}

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            tag = self._by_topic.get(message.topic)
            if tag is None:
                return
            series = samples[tag.name]
            if len(series) < count:
                series.append(self.decode_payload(tag, message.payload))

        active.on_message = on_message
        active.connect(
            self.config.broker_host, self.config.broker_port, self.config.keepalive_s
        )
        for tag in self.config.tags:
            active.subscribe(tag.topic)
        active.loop_start()
        try:
            start = time.monotonic()
            while time.monotonic() - start < deadline:
                if all(len(series) >= count for series in samples.values()):
                    break
                time.sleep(0.01)
        finally:
            active.loop_stop()
            active.disconnect()
        return samples

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the bridge configuration.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping with the configuration and the
            ``paho-mqtt`` availability flag.
        """
        return {
            "config": self.config.to_audit_record(),
            "paho_mqtt_available": HAS_PAHO_MQTT,
        }
