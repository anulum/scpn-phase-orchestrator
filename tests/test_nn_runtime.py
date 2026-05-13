# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn runtime API

from __future__ import annotations

import importlib.util

import pytest


def test_nn_exports_runtime_api() -> None:
    import scpn_phase_orchestrator.nn as nn

    expected = {
        "HAS_JAX",
        "JaxRuntimeInfo",
        "jax_runtime_info",
        "require_jax",
        "require_accelerator",
        "default_device",
    }

    assert expected.issubset(set(nn.__all__))
    assert isinstance(nn.HAS_JAX, bool)
    assert (importlib.util.find_spec("jax") is not None) == nn.HAS_JAX


def test_jax_runtime_info_is_import_safe() -> None:
    from scpn_phase_orchestrator.nn import HAS_JAX, jax_runtime_info

    info = jax_runtime_info()

    assert info.has_jax is HAS_JAX
    if HAS_JAX:
        assert info.backend
        assert info.devices
        assert info.device_count >= 1
        assert info.default_device in info.devices
    else:
        assert info.backend is None
        assert info.devices == ()
        assert info.device_count == 0
        assert info.default_device is None


def test_require_jax_returns_jax_module_when_available() -> None:
    from scpn_phase_orchestrator.nn import HAS_JAX, require_jax

    if not HAS_JAX:
        with pytest.raises(
            RuntimeError,
            match=r"pip install scpn-phase-orchestrator\[nn\]",
        ):
            require_jax()
        return

    jax = require_jax()
    assert hasattr(jax, "devices")


def test_require_accelerator_is_explicit_about_cpu_only_runtime() -> None:
    from scpn_phase_orchestrator.nn import jax_runtime_info, require_accelerator

    info = jax_runtime_info()

    if not info.has_jax:
        with pytest.raises(RuntimeError, match=r"JAX is required"):
            require_accelerator()
        return

    accelerator_devices = tuple(
        device for device in info.devices if not device.lower().startswith("cpu")
    )
    if accelerator_devices:
        assert require_accelerator() in accelerator_devices
        return

    with pytest.raises(RuntimeError, match="No JAX GPU/TPU accelerator"):
        require_accelerator()


def test_require_accelerator_allows_cpu_only_when_requested() -> None:
    from scpn_phase_orchestrator.nn import HAS_JAX, require_accelerator

    if not HAS_JAX:
        pytest.skip("JAX not installed")

    device = require_accelerator(allow_cpu=True)
    assert isinstance(device, str)
    assert device
