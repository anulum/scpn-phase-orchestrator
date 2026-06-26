# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO live feed envelope

"""Build the SPO live Studio feed using the shared control-feed envelope."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

FEED_SCHEMA = "studio.control-feed.v1"
STUDIO_ID = "scpn-phase-orchestrator"
RUNTIME_SCHEMA = "spo.studio-runtime-snapshot.v1"


@dataclass(frozen=True, slots=True)
class StudioFeedVerb:
    """A verb summary in the ``studio.control-feed.v1`` wire envelope.

    Attributes
    ----------
    name : str
        Verb name rendered by Studio.
    safety_tier : str
        Studio safety tier: ``research``, ``certified``, or ``production``.
    side_effect : str
        Studio side-effect class: ``read-only``, ``simulated``, or
        ``live-hardware``.
    timing_class : str
        Studio timing class: ``batch``, ``interactive``, or ``realtime``.
    domain_distinctive : bool
        Whether the verb is domain-specific rather than core-spine.
    """

    name: str
    safety_tier: str
    side_effect: str
    timing_class: str
    domain_distinctive: bool

    def to_record(self) -> dict[str, object]:
        """Return the verb as a JSON-safe feed record.

        Returns
        -------
        dict[str, object]
            The verb record emitted in the feed's ``verbs`` list.
        """
        return {
            "name": self.name,
            "safety_tier": self.safety_tier,
            "side_effect": self.side_effect,
            "timing_class": self.timing_class,
            "domain_distinctive": self.domain_distinctive,
        }


@dataclass(frozen=True, slots=True)
class StudioFeedClaim:
    """A claim summary in the ``studio.control-feed.v1`` wire envelope.

    Attributes
    ----------
    schema : str
        Evidence schema identifier.
    status : str
        Claim-boundary status from the shared Studio lattice.
    admission : str
        Runtime admission decision, ``admitted`` or ``rejected``.
    kind : str
        Evidence modality, such as ``measured`` or ``curated``.
    """

    schema: str
    status: str
    admission: str
    kind: str

    def to_record(self) -> dict[str, str]:
        """Return the claim as a JSON-safe feed record.

        Returns
        -------
        dict[str, str]
            The claim record emitted in the feed's ``claims`` list.
        """
        return {
            "schema": self.schema,
            "status": self.status,
            "admission": self.admission,
            "kind": self.kind,
        }


_VERBS: tuple[StudioFeedVerb, ...] = (
    StudioFeedVerb("bind", "research", "read-only", "batch", False),
    StudioFeedVerb("simulate", "research", "simulated", "batch", False),
    StudioFeedVerb("analyse", "research", "read-only", "batch", False),
    StudioFeedVerb("supervise", "research", "read-only", "batch", True),
    StudioFeedVerb("replay", "research", "read-only", "batch", False),
    StudioFeedVerb("assure", "research", "read-only", "batch", True),
)


def build_studio_control_feed(
    snapshot: Mapping[str, object],
    *,
    studio_version: str,
) -> dict[str, object]:
    """Build a live Studio feed from a runtime server snapshot.

    The envelope mirrors the sibling CONTROL ``studio.control-feed.v1`` shape
    (``feed_schema``, ``studio``, ``studio_version``, ``content_digest``,
    ``verbs``, ``claims``) and adds a SPO-specific ``runtime`` payload for live
    state ingestion. The extra field is additive: existing Studio loaders that
    only consume verbs and claims can ignore it.

    Parameters
    ----------
    snapshot : Mapping[str, object]
        JSON-safe snapshot from ``runtime.server.SimulationState.snapshot``.
    studio_version : str
        SPO package version stamped into the feed.

    Returns
    -------
    dict[str, object]
        JSON-safe Studio feed document.
    """
    runtime = runtime_summary(snapshot)
    claims = claim_summaries(runtime)
    verbs = [verb.to_record() for verb in _VERBS]
    claim_records = [claim.to_record() for claim in claims]
    return {
        "feed_schema": FEED_SCHEMA,
        "studio": STUDIO_ID,
        "studio_version": studio_version,
        "content_digest": _content_digest(verbs, claim_records),
        "verbs": verbs,
        "claims": claim_records,
        "runtime": runtime,
    }


def runtime_summary(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Return the SPO runtime payload embedded in the live Studio feed.

    Parameters
    ----------
    snapshot : Mapping[str, object]
        Server snapshot to validate and reduce.

    Returns
    -------
    dict[str, object]
        JSON-safe runtime summary with finite numeric fields.
    """
    result: dict[str, object] = {
        "schema": RUNTIME_SCHEMA,
        "step": _required_int(snapshot, "step"),
        "r_global": _required_unit_float(snapshot, "R_global"),
        "regime": _required_text(snapshot, "regime"),
        "n_oscillators": _required_int(snapshot, "n_oscillators"),
        "amplitude_mode": _required_bool(snapshot, "amplitude_mode"),
        "layers": _layer_records(snapshot.get("layers")),
    }
    if snapshot.get("mean_amplitude") is not None:
        result["mean_amplitude"] = _required_float(snapshot, "mean_amplitude")
    return result


def claim_summaries(runtime: Mapping[str, object]) -> tuple[StudioFeedClaim, ...]:
    """Return claim summaries derived from a validated runtime payload.

    Parameters
    ----------
    runtime : Mapping[str, object]
        Validated ``spo.studio-runtime-snapshot.v1`` payload.

    Returns
    -------
    tuple[StudioFeedClaim, ...]
        Claim-boundary summaries for Studio honesty rendering.

    Raises
    ------
    ValueError
        If the runtime payload has malformed coherence or layer fields.
    """
    r_global = _required_unit_float(runtime, "r_global")
    layers = runtime.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise ValueError("runtime layers must be a sequence")
    admission = "admitted" if layers else "rejected"
    coherence_status = "bounded-support" if 0.0 <= r_global <= 1.0 else "validation-gap"
    return (
        StudioFeedClaim("spo.runtime-state.v1", "bounded-model", admission, "measured"),
        StudioFeedClaim(
            "spo.phase-coherence.v1",
            coherence_status,
            admission,
            "measured",
        ),
        StudioFeedClaim("spo.regime-state.v1", "bounded-model", admission, "curated"),
    )


def render_studio_control_feed_json(
    snapshot: Mapping[str, object],
    *,
    studio_version: str,
) -> str:
    """Render a live Studio feed as deterministic JSON.

    Parameters
    ----------
    snapshot : Mapping[str, object]
        Runtime server snapshot.
    studio_version : str
        SPO package version stamped into the feed.

    Returns
    -------
    str
        Sorted, indented JSON feed with a trailing newline.
    """
    return (
        json.dumps(
            build_studio_control_feed(snapshot, studio_version=studio_version),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _content_digest(
    verbs: Sequence[Mapping[str, object]],
    claims: Sequence[Mapping[str, object]],
) -> str:
    """Return the content digest for the feed contract fields."""
    payload = json.dumps(
        {"claims": list(claims), "verbs": list(verbs)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_text(payload: Mapping[str, object], key: str) -> str:
    """Return a required non-empty text field, else raise."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    """Return a required boolean field, else raise."""
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    """Return a required non-negative integer field, else raise."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _required_float(payload: Mapping[str, object], key: str) -> float:
    """Return a required finite real field, else raise."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _required_unit_float(payload: Mapping[str, object], key: str) -> float:
    """Return a required finite unit-interval field, else raise."""
    result = _required_float(payload, key)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return result


def _layer_records(value: object) -> list[dict[str, object]]:
    """Return validated layer records for the runtime feed."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("layers must be a sequence")
    records: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"layers[{index}] must be a mapping")
        records.append(
            {
                "name": _required_text(item, "name"),
                "r": _required_unit_float(item, "R"),
                "psi": _required_float(item, "psi"),
            }
        )
    if not records:
        raise ValueError("layers must not be empty")
    return records
