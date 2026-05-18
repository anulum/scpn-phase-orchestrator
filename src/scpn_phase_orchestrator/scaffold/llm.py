# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LLM-guided domainpack scaffolding

"""LLM-guided domainpack scaffolding with strict validation and provenance.

This module converts natural-language intent into a reviewable binding-spec
proposal through a JSON-only provider contract. Live providers fail closed
unless endpoint and model configuration are present, responses must parse as one
strict JSON object, and normalized oscillators, coupling, boundaries, actuators,
periods, and safety tiers are validated before YAML is returned. The generated
proposal records source and response hashes for audit; it never installs or
activates the proposed domainpack.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.binding.types import (
    VALID_EXTRACTORS,
    VALID_KNOBS,
    VALID_SAFETY_TIERS,
    VALID_SEVERITIES,
    is_valid_channel_id,
)
from scpn_phase_orchestrator.exceptions import BindingError

JsonMap = Mapping[str, Any]

__all__ = [
    "LLMScaffoldConfig",
    "LLMScaffoldProposal",
    "LLMScaffoldProvider",
    "LocalHTTPScaffoldProvider",
    "StaticJSONScaffoldProvider",
    "configured_llm_scaffold_provider",
    "propose_domainpack_from_description",
]

_DOMAIN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class LLMScaffoldProvider(Protocol):
    """Provider interface for a JSON-only scaffold completion backend."""

    @property
    def name(self) -> str:
        """Stable provider name for audit records."""

    def complete(self, prompt: str) -> str:
        """Return a JSON object string matching the scaffold proposal contract."""


@dataclass(frozen=True)
class LLMScaffoldConfig:
    """Runtime limits for LLM-guided scaffold generation."""

    timeout_s: float = 30.0
    max_oscillators: int = 128
    max_description_chars: int = 8000
    default_sample_period_s: float = 1.0
    default_control_period_s: float = 10.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0 or not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be positive and finite")
        if self.max_oscillators <= 0:
            raise ValueError("max_oscillators must be positive")
        if self.max_description_chars <= 0:
            raise ValueError("max_description_chars must be positive")
        _finite_positive(
            self.default_sample_period_s,
            "default_sample_period_s",
        )
        _finite_positive(
            self.default_control_period_s,
            "default_control_period_s",
        )


@dataclass(frozen=True)
class LLMScaffoldProposal:
    """Validated scaffold proposal emitted by an LLM provider."""

    yaml_text: str
    validation_errors: tuple[str, ...]
    provenance: dict[str, Any]
    raw_response_sha256: str
    description_sha256: str

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-safe audit record for generated domainpack review."""
        return {
            "kind": "llm_scaffold",
            "provider": self.provenance["provider"],
            "input_family": "llm_scaffold",
            "description_sha256": self.description_sha256,
            "raw_response_sha256": self.raw_response_sha256,
            "validation_errors": list(self.validation_errors),
            "oscillator_count": self.provenance["oscillator_count"],
            "channels": list(self.provenance["channels"]),
            "sample_period_s": self.provenance["sample_period_s"],
            "control_period_s": self.provenance["control_period_s"],
        }


@dataclass(frozen=True)
class StaticJSONScaffoldProvider:
    """Offline provider used for deterministic tests and operator review files."""

    response_json: str
    provider_name: str = "static-json"

    @property
    def name(self) -> str:
        return self.provider_name

    def complete(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must be non-empty")
        return self.response_json


@dataclass(frozen=True)
class LocalHTTPScaffoldProvider:
    """HTTP provider for local or private chat-completions-compatible gateways."""

    endpoint: str
    model: str
    api_key: str | None = None
    timeout_s: float = 30.0
    provider_name: str = "local-http"

    @property
    def name(self) -> str:
        return self.provider_name

    def complete(self, prompt: str) -> str:
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        if not self.model:
            raise ValueError("model must be non-empty")
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("endpoint scheme must be http or https")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only one strict JSON object for an SCPN "
                        "domainpack scaffold. Do not include prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            # Endpoint scheme is constrained to http/https before request creation.
            with urllib.request.urlopen(  # nosec B310
                request,
                timeout=self.timeout_s,
            ) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM scaffold provider request failed: {exc}") from exc
        return _extract_completion_text(json.loads(raw))


def configured_llm_scaffold_provider(
    *,
    config: LLMScaffoldConfig | None = None,
) -> LLMScaffoldProvider:
    """Build the configured live provider or fail closed."""
    cfg = config or LLMScaffoldConfig()
    endpoint = os.environ.get("SPO_LLM_ENDPOINT", "").strip()
    model = os.environ.get("SPO_LLM_MODEL", "").strip()
    if not endpoint or not model:
        raise RuntimeError(
            "LLM scaffold provider is not configured: set SPO_LLM_ENDPOINT "
            "and SPO_LLM_MODEL, or pass --llm-response-json for offline review."
        )
    return LocalHTTPScaffoldProvider(
        endpoint=endpoint,
        model=model,
        api_key=os.environ.get("SPO_LLM_API_KEY"),
        timeout_s=cfg.timeout_s,
    )


def propose_domainpack_from_description(
    description: str,
    *,
    project_name: str,
    provider: LLMScaffoldProvider,
    config: LLMScaffoldConfig | None = None,
) -> LLMScaffoldProposal:
    """Generate and validate a domainpack scaffold from natural language intent."""
    cfg = config or LLMScaffoldConfig()
    if not _DOMAIN_NAME_RE.match(project_name):
        raise ValueError(
            f"project_name must match [a-zA-Z0-9_-]+, got {project_name!r}"
        )
    description = description.strip()
    if not description:
        raise ValueError("description must be non-empty")
    if len(description) > cfg.max_description_chars:
        raise ValueError(f"description exceeds {cfg.max_description_chars} characters")

    prompt = _scaffold_prompt(description, project_name=project_name, config=cfg)
    raw_response = provider.complete(prompt)
    payload = _parse_strict_json_object(raw_response)
    normalised = _normalise_payload(
        payload,
        project_name=project_name,
        config=cfg,
    )
    yaml_text = _binding_yaml(normalised)
    validation_errors = _validation_errors(yaml_text)
    if validation_errors:
        raise ValueError(
            "LLM scaffold generated an invalid binding spec: "
            + "; ".join(validation_errors)
        )
    channels = tuple(
        str(oscillator["channel"]) for oscillator in normalised["oscillators"]
    )
    return LLMScaffoldProposal(
        yaml_text=yaml_text,
        validation_errors=validation_errors,
        provenance={
            "input_family": "llm_scaffold",
            "provider": provider.name,
            "oscillator_count": len(normalised["oscillators"]),
            "channels": channels,
            "sample_period_s": normalised["sample_period_s"],
            "control_period_s": normalised["control_period_s"],
            "validator": "load_binding_spec+validate_binding_spec",
        },
        raw_response_sha256=_sha256_text(raw_response),
        description_sha256=_sha256_text(description),
    )


def _scaffold_prompt(
    description: str,
    *,
    project_name: str,
    config: LLMScaffoldConfig,
) -> str:
    return "\n".join(
        [
            "Create one SCPN domainpack scaffold proposal as strict JSON.",
            f"Project name: {project_name}",
            f"Maximum oscillators: {config.max_oscillators}",
            "Allowed safety_tier values: research, clinical, consumer, production.",
            "Allowed channel identifiers include P, I, S, or named identifiers.",
            "Allowed extractor_type values: physical, informational, symbolic, "
            "hilbert, wavelet, zero_crossing, event, ring, graph.",
            "Required JSON keys: name, oscillators.",
            "Optional JSON keys: sample_period_s, control_period_s, safety_tier, "
            "coupling, boundaries, actuators.",
            "Each oscillator requires id, channel, extractor_type; omega is optional.",
            "Return no markdown and no explanatory text.",
            "",
            "Description:",
            description,
        ]
    )


def _parse_strict_json_object(raw_response: str) -> JsonMap:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM scaffold response must be a strict JSON object") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("LLM scaffold response must be a JSON object")
    return payload


def _normalise_payload(
    payload: JsonMap,
    *,
    project_name: str,
    config: LLMScaffoldConfig,
) -> dict[str, Any]:
    raw_name = payload.get("name", project_name)
    if not isinstance(raw_name, str) or not _DOMAIN_NAME_RE.match(raw_name):
        raise ValueError("proposal name must match [a-zA-Z0-9_-]+")
    if raw_name != project_name:
        raise ValueError("proposal name must match requested project_name")
    sample_period_s = _optional_positive_number(
        payload.get("sample_period_s"),
        "sample_period_s",
        default=config.default_sample_period_s,
    )
    control_period_s = _optional_positive_number(
        payload.get("control_period_s"),
        "control_period_s",
        default=max(config.default_control_period_s, sample_period_s),
    )
    if control_period_s < sample_period_s:
        raise ValueError("control_period_s must be >= sample_period_s")
    safety_tier = payload.get("safety_tier", "research")
    if not isinstance(safety_tier, str) or safety_tier not in VALID_SAFETY_TIERS:
        raise ValueError(f"safety_tier must be one of {sorted(VALID_SAFETY_TIERS)}")

    oscillators = _normalise_oscillators(
        payload.get("oscillators"),
        max_oscillators=config.max_oscillators,
    )
    coupling = _normalise_coupling(payload.get("coupling"))
    boundaries = _normalise_boundaries(payload.get("boundaries", ()))
    actuators = _normalise_actuators(payload.get("actuators", ()))
    return {
        "name": raw_name,
        "sample_period_s": sample_period_s,
        "control_period_s": control_period_s,
        "safety_tier": safety_tier,
        "oscillators": oscillators,
        "coupling": coupling,
        "boundaries": boundaries,
        "actuators": actuators,
    }


def _normalise_oscillators(
    raw_oscillators: object,
    *,
    max_oscillators: int,
) -> tuple[dict[str, Any], ...]:
    values = _sequence(raw_oscillators, "oscillators")
    if not values:
        raise ValueError("proposal must include at least one oscillator")
    if len(values) > max_oscillators:
        raise ValueError(f"proposal exceeds {max_oscillators} oscillators")
    seen_ids: set[str] = set()
    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        item = _mapping(raw, f"oscillators[{index}]")
        oscillator_id = _identifier(item.get("id"), f"oscillators[{index}].id")
        if oscillator_id in seen_ids:
            raise ValueError(f"duplicate oscillator id: {oscillator_id}")
        seen_ids.add(oscillator_id)
        channel = _identifier(item.get("channel"), f"oscillators[{index}].channel")
        if not is_valid_channel_id(channel):
            raise ValueError(f"invalid oscillator channel: {channel!r}")
        extractor_type = _string(
            item.get("extractor_type"),
            f"oscillators[{index}].extractor_type",
        )
        if extractor_type not in VALID_EXTRACTORS:
            raise ValueError(
                f"oscillator extractor_type must be one of {sorted(VALID_EXTRACTORS)}"
            )
        omega = _optional_finite_number(
            item.get("omega"),
            f"oscillators[{index}].omega",
            default=1.0,
        )
        normalised.append(
            {
                "id": oscillator_id,
                "channel": channel,
                "extractor_type": extractor_type,
                "omega": omega,
            }
        )
    return tuple(normalised)


def _normalise_coupling(raw_coupling: object) -> dict[str, float]:
    if raw_coupling is None:
        return {"base_strength": 0.45, "decay_alpha": 0.3}
    coupling = _mapping(raw_coupling, "coupling")
    base_strength = _optional_non_negative_number(
        coupling.get("base_strength"),
        "coupling.base_strength",
        default=0.45,
    )
    decay_alpha = _optional_non_negative_number(
        coupling.get("decay_alpha"),
        "coupling.decay_alpha",
        default=0.3,
    )
    return {"base_strength": base_strength, "decay_alpha": decay_alpha}


def _normalise_boundaries(raw_boundaries: object) -> tuple[dict[str, Any], ...]:
    values = _sequence(raw_boundaries, "boundaries")
    normalised = []
    for index, raw in enumerate(values):
        item = _mapping(raw, f"boundaries[{index}]")
        severity = _string(item.get("severity"), f"boundaries[{index}].severity")
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"boundary severity must be one of {sorted(VALID_SEVERITIES)}"
            )
        lower = _optional_nullable_finite_number(
            item.get("lower"),
            f"boundaries[{index}].lower",
        )
        upper = _optional_nullable_finite_number(
            item.get("upper"),
            f"boundaries[{index}].upper",
        )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("boundary lower must be < upper")
        normalised.append(
            {
                "name": _identifier(item.get("name"), f"boundaries[{index}].name"),
                "variable": _identifier(
                    item.get("variable"),
                    f"boundaries[{index}].variable",
                ),
                "lower": lower,
                "upper": upper,
                "severity": severity,
            }
        )
    return tuple(normalised)


def _normalise_actuators(raw_actuators: object) -> tuple[dict[str, Any], ...]:
    values = _sequence(raw_actuators, "actuators")
    normalised = []
    for index, raw in enumerate(values):
        item = _mapping(raw, f"actuators[{index}]")
        knob = _string(item.get("knob"), f"actuators[{index}].knob")
        if knob not in VALID_KNOBS:
            raise ValueError(f"actuator knob must be one of {sorted(VALID_KNOBS)}")
        limits = _number_pair(item.get("limits"), f"actuators[{index}].limits")
        if limits[0] > limits[1]:
            raise ValueError("actuator limits must satisfy lower <= upper")
        normalised.append(
            {
                "name": _identifier(item.get("name"), f"actuators[{index}].name"),
                "knob": knob,
                "scope": _string(item.get("scope"), f"actuators[{index}].scope"),
                "limits": limits,
            }
        )
    return tuple(normalised)


def _binding_yaml(payload: Mapping[str, Any]) -> str:
    layer_lines = []
    family_lines = []
    good_layers = []
    for index, oscillator in enumerate(payload["oscillators"]):
        family_name = f"llm_{index}_{oscillator['channel'].lower()}"
        good_layers.append(str(index))
        layer_lines.extend(
            [
                f"  - name: {_yaml_scalar(oscillator['id'])}",
                f"    index: {index}",
                f"    oscillator_ids: [{_yaml_scalar(oscillator['id'])}]",
                f"    omegas: [{float(oscillator['omega']):.12g}]",
                f"    family: {_yaml_scalar(family_name)}",
            ]
        )
        family_lines.extend(
            [
                f"  {family_name}:",
                f"    channel: {_yaml_scalar(oscillator['channel'])}",
                f"    extractor_type: {_yaml_scalar(oscillator['extractor_type'])}",
                "    config: {}",
            ]
        )
    boundary_lines = _mapping_list_yaml(payload["boundaries"], indent=2)
    actuator_lines = _mapping_list_yaml(payload["actuators"], indent=2)
    coupling = payload["coupling"]
    return "\n".join(
        [
            "# LLM-assisted binding proposal. Review before production use.",
            f"name: {_yaml_scalar(payload['name'])}",
            'version: "0.1.0"',
            f"safety_tier: {_yaml_scalar(payload['safety_tier'])}",
            f"sample_period_s: {float(payload['sample_period_s']):.12g}",
            f"control_period_s: {float(payload['control_period_s']):.12g}",
            "",
            "layers:",
            *layer_lines,
            "",
            "oscillator_families:",
            *family_lines,
            "",
            "coupling:",
            f"  base_strength: {coupling['base_strength']:.12g}",
            f"  decay_alpha: {coupling['decay_alpha']:.12g}",
            "  templates: {}",
            "",
            "drivers:",
            "  physical:",
            "    zeta: 0.0",
            "    psi: 0.0",
            "  informational:",
            "    zeta: 0.02",
            "  symbolic:",
            "    zeta: 0.02",
            "",
            "objectives:",
            f"  good_layers: [{', '.join(good_layers)}]",
            "  bad_layers: []",
            "  good_weight: 1.0",
            "  bad_weight: 1.0",
            "",
            "boundaries:",
            *(boundary_lines or ["  []"]),
            "",
            "actuators:",
            *(actuator_lines or ["  []"]),
            "",
            "amplitude:",
            "  mu: 1.0",
            "  epsilon: 0.3",
            "  amp_coupling_strength: 0.2",
            "  amp_coupling_decay: 0.3",
            "",
            "policy: policy.yaml",
            "",
        ]
    )


def _mapping_list_yaml(items: Sequence[Mapping[str, Any]], *, indent: int) -> list[str]:
    if not items:
        return []
    prefix = " " * indent
    continuation = " " * (indent + 2)
    lines: list[str] = []
    for item in items:
        first = True
        for key, value in item.items():
            value_text = _yaml_value(value)
            if first:
                lines.append(f"{prefix}- {key}: {value_text}")
                first = False
            else:
                lines.append(f"{continuation}{key}: {value_text}")
    return lines


def _yaml_value(value: object) -> str:
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_yaml_value(item) for item in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, str):
        return _yaml_scalar(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return f"{float(value):.12g}"
    raise TypeError(f"unsupported YAML value: {type(value).__name__}")


def _yaml_scalar(value: str) -> str:
    return json.dumps(value)


def _validation_errors(yaml_text: str) -> tuple[str, ...]:
    with tempfile.TemporaryDirectory(prefix="spo_llm_scaffold_") as tmpdir:
        path = Path(tmpdir) / "binding_spec.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        try:
            spec = load_binding_spec(path)
        except (BindingError, ValueError) as exc:
            return (str(exc),)
        return tuple(validate_binding_spec(spec))


def _extract_completion_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("provider response must be a JSON mapping or string")
    if isinstance(payload.get("text"), str):
        return str(payload["text"])
    if isinstance(payload.get("content"), str):
        return str(payload["content"])
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, str) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return str(message["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
    raise ValueError("provider response does not contain completion text")


def _mapping(value: object, label: str) -> JsonMap:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, label: str) -> str:
    text = _string(value, label)
    if not _DOMAIN_NAME_RE.match(text):
        raise ValueError(f"{label} must match [a-zA-Z0-9_-]+")
    return text


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _number_pair(value: object, label: str) -> tuple[float, float]:
    values = _sequence(value, label)
    if len(values) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    return (
        _finite_number(values[0], f"{label}[0]"),
        _finite_number(values[1], f"{label}[1]"),
    )


def _optional_positive_number(value: object, label: str, *, default: float) -> float:
    if value is None:
        return default
    number = _finite_number(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _optional_non_negative_number(
    value: object, label: str, *, default: float
) -> float:
    if value is None:
        return default
    number = _finite_number(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _optional_finite_number(value: object, label: str, *, default: float) -> float:
    if value is None:
        return default
    return _finite_number(value, label)


def _optional_nullable_finite_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label)


def _finite_positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
