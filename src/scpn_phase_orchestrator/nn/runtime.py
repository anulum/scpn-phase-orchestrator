# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JAX runtime contract for nn/

"""Runtime contract for the GPU-first differentiable ``nn`` API.

The numerical layers remain pure JAX/equinox modules. This module makes the
runtime status explicit for ML users: whether JAX is installed, which backend
is active, which devices are visible, and whether a non-CPU accelerator is
available for production training paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from types import ModuleType

HAS_JAX = util.find_spec("jax") is not None


@dataclass(frozen=True, slots=True)
class JaxRuntimeInfo:
    """Snapshot of the active JAX runtime visible to ``nn`` callers."""

    has_jax: bool
    backend: str | None
    devices: tuple[str, ...]
    default_device: str | None
    device_count: int
    accelerator_count: int

    @property
    def has_accelerator(self) -> bool:
        """Return ``True`` when at least one non-CPU JAX device is visible."""
        return self.accelerator_count > 0


def require_jax() -> ModuleType:
    """Return the imported JAX module or raise a clear installation error."""
    if not HAS_JAX:
        msg = (
            "JAX is required for scpn_phase_orchestrator.nn. "
            "Install with: pip install scpn-phase-orchestrator[nn]"
        )
        raise RuntimeError(msg)

    import jax

    return jax


def _device_kind(device: object) -> str:
    platform = getattr(device, "platform", None)
    if isinstance(platform, str) and platform:
        return platform.lower()
    kind = getattr(device, "device_kind", None)
    if isinstance(kind, str) and kind:
        return kind.lower()
    return str(device).lower()


def _device_label(device: object) -> str:
    kind = _device_kind(device)
    identifier = getattr(device, "id", None)
    if isinstance(identifier, int):
        return f"{kind}:{identifier}"
    return kind


def jax_runtime_info() -> JaxRuntimeInfo:
    """Return an import-safe summary of the active JAX runtime.

    The function never imports JAX when it is not installed, so base package
    users can still import ``scpn_phase_orchestrator.nn`` and receive an
    actionable runtime report.
    """
    if not HAS_JAX:
        return JaxRuntimeInfo(
            has_jax=False,
            backend=None,
            devices=(),
            default_device=None,
            device_count=0,
            accelerator_count=0,
        )

    jax = require_jax()
    devices_raw = tuple(jax.devices())
    devices = tuple(_device_label(device) for device in devices_raw)
    accelerators = tuple(
        device for device in devices_raw if not _device_kind(device).startswith("cpu")
    )
    default_backend = str(jax.default_backend())
    default_device = devices[0] if devices else None
    return JaxRuntimeInfo(
        has_jax=True,
        backend=default_backend,
        devices=devices,
        default_device=default_device,
        device_count=len(devices),
        accelerator_count=len(accelerators),
    )


def default_device() -> str:
    """Return the default JAX device label used by the ``nn`` API."""
    info = jax_runtime_info()
    if info.default_device is None:
        msg = (
            "JAX is required for scpn_phase_orchestrator.nn. "
            "Install with: pip install scpn-phase-orchestrator[nn]"
        )
        raise RuntimeError(msg)
    return info.default_device


def require_accelerator(*, allow_cpu: bool = False) -> str:
    """Return the production training device or fail fast on CPU-only runtimes.

    Args:
        allow_cpu: Permit CPU-only JAX execution. This is useful for CI,
            documentation examples, and small smoke tests. Production ML
            training should keep the default ``False`` so misconfigured GPU
            jobs fail before expensive work starts.

    Returns:
        A JAX device label such as ``"gpu:0"``, ``"tpu:0"``, or ``"cpu:0"``
        when ``allow_cpu`` is enabled.
    """
    info = jax_runtime_info()
    if not info.has_jax:
        msg = (
            "JAX is required for scpn_phase_orchestrator.nn. "
            "Install with: pip install scpn-phase-orchestrator[nn]"
        )
        raise RuntimeError(msg)

    accelerator = next(
        (device for device in info.devices if not device.lower().startswith("cpu")),
        None,
    )
    if accelerator is not None:
        return accelerator
    if allow_cpu and info.default_device is not None:
        return info.default_device

    msg = (
        "No JAX GPU/TPU accelerator is visible. Install a hardware-enabled "
        "jaxlib build or configure the accelerator runtime before production "
        "training. For CI or smoke tests, call require_accelerator(allow_cpu=True)."
    )
    raise RuntimeError(msg)
