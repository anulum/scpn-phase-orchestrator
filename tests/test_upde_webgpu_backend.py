# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WebGPU UPDE backend tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_webgpu import (
    WEBGPU_WORKGROUP_SIZE,
    WebGPUBackendUnavailableError,
    build_webgpu_upde_package,
    get_webgpu_backend_capabilities,
    is_webgpu_runtime_available,
    upde_run_webgpu,
)
from scpn_phase_orchestrator.upde import _run as run_mod


def test_webgpu_capabilities_are_explicit_about_precision_and_methods() -> None:
    caps = get_webgpu_backend_capabilities()

    assert caps.name == "webgpu"
    assert caps.execution_target == "browser-or-edge-webgpu"
    assert caps.supported_methods == ("euler",)
    assert caps.scalar_type == "f32"
    assert caps.workgroup_size == WEBGPU_WORKGROUP_SIZE
    assert "dense Kuramoto" in caps.numerical_contract


def test_webgpu_package_contains_real_wgsl_and_runner_sources() -> None:
    package = build_webgpu_upde_package(method="euler")

    assert package.method == "euler"
    assert "@compute @workgroup_size(64)" in package.wgsl
    assert "var<storage, read> phases_in" in package.wgsl
    assert "var<storage, read_write> phases_out" in package.wgsl
    assert "sin(phases_in[j] - theta - alpha[idx])" in package.wgsl
    assert "params.zeta * sin(params.psi - theta)" in package.wgsl
    assert "dispatchWorkgroups" in package.javascript
    assert "GPUBufferUsage.STORAGE" in package.javascript
    assert "makeRawBuffer(" in package.javascript
    assert "params,\n      GPUBufferUsage.UNIFORM" in package.javascript
    assert "new Uint8Array(buffer.getMappedRange())" in package.javascript
    assert "navigator.gpu" in package.javascript
    assert "TODO" not in package.wgsl
    assert "TODO" not in package.javascript


def test_webgpu_package_rejects_unsupported_integrators() -> None:
    with pytest.raises(ValueError, match="WebGPU UPDE backend currently supports"):
        build_webgpu_upde_package(method="rk4")


def test_webgpu_runtime_is_not_reported_available_on_plain_cpython() -> None:
    assert is_webgpu_runtime_available() is False
    with pytest.raises(WebGPUBackendUnavailableError, match="browser"):
        upde_run_webgpu()


def test_dispatcher_declares_webgpu_loader_without_making_it_default() -> None:
    assert "webgpu" in run_mod._BACKEND_NAMES
    assert "webgpu" in run_mod._LOADERS
    with pytest.raises(RuntimeError, match="bridge is not configured"):
        run_mod._LOADERS["webgpu"]()
    assert "python" in run_mod.AVAILABLE_BACKENDS
    assert run_mod.AVAILABLE_BACKENDS[0] == run_mod.ACTIVE_BACKEND
