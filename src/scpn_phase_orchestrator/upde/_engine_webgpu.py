# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WebGPU UPDE backend package

"""Browser and edge WebGPU backend assets for UPDE integration.

The CPython dispatcher cannot execute browser WebGPU directly. This module
therefore owns two separate contracts:

* capability detection for Pyodide/browser-like runtimes where ``navigator.gpu``
  is reachable through the JavaScript bridge;
* reproducible WGSL and ES-module source generation for the browser/edge
  backend package.

The shader implements the same dense Euler Sakaguchi-Kuramoto derivative as the
NumPy reference path:

``dtheta_i = omega_i + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
             + zeta sin(psi - theta_i)``

The browser kernel uses WGSL ``f32`` because portable WebGPU does not expose
``f64`` arithmetic. It is therefore an interactive and edge portability path,
not a replacement for the ``f64`` Rust/JAX certification or training paths.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

WEBGPU_WORKGROUP_SIZE = 64
_SUPPORTED_METHODS = ("euler",)
_BRIDGE_ENV = "SPO_WEBGPU_DISPATCH_BRIDGE"


class WebGPUBackendUnavailableError(RuntimeError):
    """Raised when CPython is asked to execute browser-only WebGPU code."""


@dataclass(frozen=True)
class WebGPUBackendCapabilities:
    """Documented execution contract for the WebGPU UPDE backend."""

    name: str
    execution_target: str
    supported_methods: tuple[str, ...]
    scalar_type: str
    workgroup_size: int
    numerical_contract: str


@dataclass(frozen=True)
class WebGPUKernelPackage:
    """WGSL and JavaScript sources required by a browser/edge host."""

    method: str
    wgsl: str
    javascript: str
    workgroup_size: int


def get_webgpu_backend_capabilities() -> WebGPUBackendCapabilities:
    """Return the explicit WebGPU execution contract."""

    return WebGPUBackendCapabilities(
        name="webgpu",
        execution_target="browser-or-edge-webgpu",
        supported_methods=_SUPPORTED_METHODS,
        scalar_type="f32",
        workgroup_size=WEBGPU_WORKGROUP_SIZE,
        numerical_contract=(
            "dense Kuramoto Euler step with K_nm, alpha_nm, omega_i, zeta, "
            "psi, phase wrapping, and browser WebGPU f32 arithmetic"
        ),
    )


def is_webgpu_runtime_available() -> bool:
    """Detect browser WebGPU availability without importing JS on CPython."""

    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return False
    navigator = getattr(js, "navigator", None)
    return bool(navigator is not None and getattr(navigator, "gpu", None) is not None)


def _validate_method(method: str) -> str:
    if method not in _SUPPORTED_METHODS:
        supported = ", ".join(_SUPPORTED_METHODS)
        msg = f"WebGPU UPDE backend currently supports {supported}; got {method!r}"
        raise ValueError(msg)
    return method


def _webgpu_wgsl() -> str:
    return f"""struct Params {{
    n: u32,
    dt: f32,
    zeta: f32,
    psi: f32,
    two_pi: f32,
}};

@group(0) @binding(0) var<storage, read> phases_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> phases_out: array<f32>;
@group(0) @binding(2) var<storage, read> omegas: array<f32>;
@group(0) @binding(3) var<storage, read> knm: array<f32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn wrap_phase(theta: f32, two_pi: f32) -> f32 {{
    var wrapped = theta - floor(theta / two_pi) * two_pi;
    if (wrapped < 0.0) {{
        wrapped = wrapped + two_pi;
    }}
    return wrapped;
}}

@compute @workgroup_size({WEBGPU_WORKGROUP_SIZE})
fn upde_euler(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i = global_id.x;
    if (i >= params.n) {{
        return;
    }}

    let theta = phases_in[i];
    var coupling = 0.0;
    for (var j = 0u; j < params.n; j = j + 1u) {{
        let idx = i * params.n + j;
        coupling = coupling + knm[idx] * sin(phases_in[j] - theta - alpha[idx]);
    }}

    let drive = params.zeta * sin(params.psi - theta);
    let derivative = omegas[i] + coupling + drive;
    phases_out[i] = wrap_phase(theta + params.dt * derivative, params.two_pi);
}}
"""


def _webgpu_javascript() -> str:
    return """const TWO_PI = Math.PI * 2.0;
const WORKGROUP_SIZE = 64;

function assertFiniteArray(name, values, expectedLength) {
  if (!(values instanceof Float32Array)) {
    throw new TypeError(`${name} must be a Float32Array`);
  }
  if (values.length !== expectedLength) {
    throw new RangeError(`${name} length ${values.length} != ${expectedLength}`);
  }
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) {
      throw new RangeError(`${name}[${i}] is not finite`);
    }
  }
}

function makeBuffer(device, values, usage) {
  const buffer = device.createBuffer({
    size: values.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}

function makeRawBuffer(device, bytes, usage) {
  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
  buffer.unmap();
  return buffer;
}

export class WebGPUUPDEBackend {
  constructor(device, pipeline, bindGroupLayout) {
    this.device = device;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  static async create(shaderSource) {
    if (!globalThis.navigator || !navigator.gpu) {
      throw new Error("WebGPU is unavailable: navigator.gpu is missing");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU adapter request failed");
    }
    const device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: shaderSource });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shader, entryPoint: "upde_euler" },
    });
    return new WebGPUUPDEBackend(device, pipeline, pipeline.getBindGroupLayout(0));
  }

  async runEuler({
    phases, omegas, knm, alpha, zeta, psi, dt, nSteps = 1, nSubsteps = 1,
  }) {
    const n = phases.length;
    if (n < 1) {
      throw new RangeError("phases must contain at least one oscillator");
    }
    if (!Number.isInteger(nSteps) || nSteps < 1) {
      throw new RangeError("nSteps must be a positive integer");
    }
    if (!Number.isInteger(nSubsteps) || nSubsteps < 1) {
      throw new RangeError("nSubsteps must be a positive integer");
    }
    assertFiniteArray("phases", phases, n);
    assertFiniteArray("omegas", omegas, n);
    assertFiniteArray("knm", knm, n * n);
    assertFiniteArray("alpha", alpha, n * n);
    for (const [name, value] of [["zeta", zeta], ["psi", psi], ["dt", dt]]) {
      if (!Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
      }
    }

    const device = this.device;
    const usage = GPUBufferUsage.STORAGE
      | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST;
    let src = makeBuffer(device, phases, usage);
    let dst = device.createBuffer({ size: phases.byteLength, usage });
    const omegaBuffer = makeBuffer(device, omegas, GPUBufferUsage.STORAGE);
    const knmBuffer = makeBuffer(device, knm, GPUBufferUsage.STORAGE);
    const alphaBuffer = makeBuffer(device, alpha, GPUBufferUsage.STORAGE);
    const params = new ArrayBuffer(32);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = n;
    paramsF32[1] = dt / nSubsteps;
    paramsF32[2] = zeta;
    paramsF32[3] = psi;
    paramsF32[4] = TWO_PI;
    const paramsBuffer = makeRawBuffer(
      device,
      params,
      GPUBufferUsage.UNIFORM,
    );

    const makeBindGroup = () => device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: omegaBuffer } },
        { binding: 3, resource: { buffer: knmBuffer } },
        { binding: 4, resource: { buffer: alphaBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });

    const totalPasses = nSteps * nSubsteps;
    const workgroups = Math.ceil(n / WORKGROUP_SIZE);
    for (let passIndex = 0; passIndex < totalPasses; passIndex += 1) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, makeBindGroup());
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      device.queue.submit([encoder.finish()]);
      [src, dst] = [dst, src];
    }

    const readBuffer = device.createBuffer({
      size: phases.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, readBuffer, 0, phases.byteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return result;
  }
}
"""


def build_webgpu_upde_package(method: str = "euler") -> WebGPUKernelPackage:
    """Build the browser WebGPU kernel package for a supported method."""

    method = _validate_method(method)
    return WebGPUKernelPackage(
        method=method,
        wgsl=_webgpu_wgsl(),
        javascript=_webgpu_javascript(),
        workgroup_size=WEBGPU_WORKGROUP_SIZE,
    )


def load_webgpu_dispatch_bridge() -> Callable[..., FloatArray]:
    """Load an explicit Python bridge to a host-managed WebGPU executor.

    Browser and edge deployments normally run the generated ES-module runner
    directly. Python dispatch can only be considered available when a host has
    installed a callable bridge and exposed it through
    ``SPO_WEBGPU_DISPATCH_BRIDGE=module:function``.
    """

    spec = os.environ.get(_BRIDGE_ENV)
    if not spec:
        raise WebGPUBackendUnavailableError(
            "WebGPU dispatcher bridge is not configured; set "
            f"{_BRIDGE_ENV}=module:function or use build_webgpu_upde_package()"
        )

    module_name, separator, attr_name = spec.partition(":")
    if not separator or not module_name or not attr_name:
        raise WebGPUBackendUnavailableError(
            f"{_BRIDGE_ENV} must use the form module:function; got {spec!r}"
        )

    try:
        bridge = getattr(importlib.import_module(module_name), attr_name)
    except (ImportError, AttributeError) as exc:
        raise WebGPUBackendUnavailableError(
            f"unable to load WebGPU dispatcher bridge {spec!r}"
        ) from exc

    if not callable(bridge):
        raise WebGPUBackendUnavailableError(
            f"WebGPU dispatcher bridge {spec!r} is not callable"
        )
    return cast("Callable[..., FloatArray]", bridge)


def upde_run_webgpu(*_args: object, **_kwargs: object) -> FloatArray:
    """CPython placeholder for dispatcher compatibility.

    Browser execution uses the generated JavaScript runner. CPython callers
    should use :func:`build_webgpu_upde_package` to export the backend package
    instead of expecting local WebGPU execution.
    """

    raise WebGPUBackendUnavailableError(
        "WebGPU execution requires a browser or edge runtime with navigator.gpu; "
        "use build_webgpu_upde_package() to export the browser backend"
    )
