# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor PPO checkpoint IO

"""Serialisation and restoration of supervisor PPO checkpoints.

A checkpoint is two files, ``state.eqx`` and ``metadata.json``, replaced one after
the other, so a crash between the two replacements leaves a new payload beside the
previous metadata. Schema version 2 records the SHA-256 of the payload in the
metadata, and loading refuses a payload whose digest differs; version 1
checkpoints carry no digest and are loaded unverified. Typed PRNG keys
(``jax.random.key``) are stored as their raw key data with the implementation
name and re-wrapped on load.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._shared import _json_object, _non_negative_int
from ._types import SupervisorPPOCheckpoint, _SupervisorPPOCheckpointPayload

if TYPE_CHECKING:
    from .policy import DifferentiableSupervisorPolicy


_SUPERVISOR_PPO_CHECKPOINT_FORMAT = (
    "scpn_phase_orchestrator.nn.supervisor.ppo_checkpoint"
)


_SUPERVISOR_PPO_CHECKPOINT_SCHEMA_VERSION = 2

#: Schema versions ``load_supervisor_ppo_checkpoint`` reads; 1 has no payload digest.
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def save_supervisor_ppo_checkpoint(
    checkpoint_dir: str | Path,
    *,
    policy: DifferentiableSupervisorPolicy,
    opt_state: Any,
    key: jax.Array,
    n_updates: int,
    loss_history: jax.Array,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist PPO supervisor trainer state for deterministic resume.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory for training checkpoints, or ``None``.
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    opt_state : Any
        The optax optimiser state.
    key : jax.Array
        JAX PRNG key.
    n_updates : int
        Number of optimiser updates performed.
    loss_history : jax.Array
        Recorded per-step loss history.
    metadata : dict[str, Any] | None
        Associated metadata mapping, or ``None``.
    overwrite : bool
        Whether to overwrite an existing checkpoint.

    Returns
    -------
    Path
        The path of the written checkpoint.

    Raises
    ------
    NotADirectoryError
        If the checkpoint directory path is not a directory.
    FileExistsError
        If the checkpoint already exists and ``overwrite`` is false.
    """
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists() and not checkpoint_path.is_dir():
        raise NotADirectoryError(f"{checkpoint_path} is not a checkpoint directory")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_path / "state.eqx"
    metadata_path = checkpoint_path / "metadata.json"
    if not overwrite and (state_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"{checkpoint_path} already contains a supervisor PPO checkpoint"
        )

    n_updates = _non_negative_int(n_updates, "n_updates")
    key = jnp.asarray(key)
    key_impl: str | None = None
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        # Equinox serialises arrays, not typed keys: keep the raw key data and
        # the implementation name needed to wrap it again.
        key_impl = str(jax.random.key_impl(key))
        key = jax.random.key_data(key)
    loss_history = jnp.asarray(loss_history)
    user_metadata = _json_object(metadata, "metadata")
    payload = _SupervisorPPOCheckpointPayload(
        policy=policy,
        opt_state=opt_state,
        key=key,
        loss_history=loss_history,
    )
    checkpoint_metadata = {
        "format": _SUPERVISOR_PPO_CHECKPOINT_FORMAT,
        "schema_version": _SUPERVISOR_PPO_CHECKPOINT_SCHEMA_VERSION,
        "n_updates": n_updates,
        "key_shape": list(key.shape),
        "key_dtype": str(key.dtype),
        "key_impl": key_impl,
        "loss_history_shape": list(loss_history.shape),
        "loss_history_dtype": str(loss_history.dtype),
        "metadata": user_metadata,
    }

    state_tmp = checkpoint_path / "state.eqx.tmp"
    metadata_tmp = checkpoint_path / "metadata.json.tmp"
    eqx.tree_serialise_leaves(state_tmp, payload)
    checkpoint_metadata["state_sha256"] = hashlib.sha256(
        state_tmp.read_bytes()
    ).hexdigest()
    metadata_tmp.write_text(
        json.dumps(checkpoint_metadata, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    state_tmp.replace(state_path)
    metadata_tmp.replace(metadata_path)
    return checkpoint_path


def load_supervisor_ppo_checkpoint(
    checkpoint_dir: str | Path,
    *,
    template_policy: DifferentiableSupervisorPolicy,
    template_opt_state: Any,
) -> SupervisorPPOCheckpoint:
    """Load a PPO supervisor checkpoint against explicit policy/state templates.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory for training checkpoints, or ``None``.
    template_policy : DifferentiableSupervisorPolicy
        Template policy used to reconstruct the checkpoint.
    template_opt_state : Any
        Template optimiser state used to reconstruct the checkpoint.

    Returns
    -------
    SupervisorPPOCheckpoint
        The loaded PPO checkpoint.

    Raises
    ------
    FileNotFoundError
        If the checkpoint cannot be found.
    ValueError
        If the metadata is malformed, or (schema version 2) the payload's SHA-256
        differs from the digest recorded in the metadata.
    """
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    state_path = checkpoint_path / "state.eqx"
    metadata = _load_checkpoint_metadata(metadata_path)
    if not state_path.exists():
        raise FileNotFoundError(f"missing checkpoint payload: {state_path}")
    state_bytes = state_path.read_bytes()
    if metadata["schema_version"] >= 2:
        expected = metadata.get("state_sha256")
        if not isinstance(expected, str) or _SHA256_HEX.fullmatch(expected) is None:
            raise ValueError("state_sha256 must be a lowercase SHA-256 hex digest")
        if hashlib.sha256(state_bytes).hexdigest() != expected:
            raise ValueError(
                "checkpoint payload does not match its metadata (state_sha256); "
                "the pair was not written together"
            )
    key_impl = metadata.get("key_impl")
    if key_impl is not None and not isinstance(key_impl, str):
        raise ValueError("key_impl must be a PRNG implementation name or null")
    n_updates = _non_negative_int(metadata.get("n_updates"), "n_updates")

    key_shape = _metadata_shape(metadata, "key_shape")
    loss_history_shape = _metadata_shape(metadata, "loss_history_shape")
    key_dtype = _metadata_dtype(metadata, "key_dtype")
    loss_history_dtype = _metadata_dtype(metadata, "loss_history_dtype")
    template_payload = _SupervisorPPOCheckpointPayload(
        policy=template_policy,
        opt_state=template_opt_state,
        key=jnp.zeros(key_shape, dtype=key_dtype),
        loss_history=jnp.zeros(loss_history_shape, dtype=loss_history_dtype),
    )
    # Deserialise the bytes that were verified, not a second read of the file.
    loaded = eqx.tree_deserialise_leaves(io.BytesIO(state_bytes), template_payload)
    key = loaded.key
    if key_impl is not None:
        try:
            key = jax.random.wrap_key_data(key, impl=key_impl)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"key_impl {key_impl!r} cannot wrap the key") from exc
    return SupervisorPPOCheckpoint(
        policy=loaded.policy,
        opt_state=loaded.opt_state,
        key=key,
        loss_history=loaded.loss_history,
        n_updates=n_updates,
        metadata=_json_object(metadata.get("metadata"), "checkpoint metadata"),
    )


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Return the validated checkpoint metadata, else raise."""
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    version = raw.get("schema_version")
    if (
        raw.get("format") != _SUPERVISOR_PPO_CHECKPOINT_FORMAT
        or isinstance(version, bool)
        or version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError("checkpoint schema is not supported")
    return raw


def _metadata_shape(metadata: dict[str, Any], field: str) -> tuple[int, ...]:
    """Return the array shape recorded in checkpoint metadata."""
    raw = metadata.get(field)
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a shape list")
    shape: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field}[{index}] must be a non-negative integer")
        shape.append(value)
    return tuple(shape)


def _metadata_dtype(metadata: dict[str, Any], field: str) -> jnp.dtype:
    """Return the array dtype recorded in checkpoint metadata."""
    raw = metadata.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a dtype string")
    try:
        return jnp.dtype(raw)
    except TypeError as exc:
        raise ValueError(f"{field} is not a supported dtype") from exc
