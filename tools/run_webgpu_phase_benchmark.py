# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Executed browser WebGPU phase benchmark

"""Measure the generated UPDE package in an owned, real Chromium process.

The caller supplies an installed Node Playwright package and browser executable.
No adapter, shader, device, or output is replaced. Adapter information identifies
software fallback execution explicitly; timings include submission and readback.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import subprocess
import tempfile
import threading
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from scpn_phase_orchestrator.upde._engine_webgpu import build_webgpu_upde_package

_DRIVER = r"""
const { chromium } = require(process.argv[2]);
(async () => {
  const browser = await chromium.launch({
    executablePath: process.argv[3], headless: true,
    args: ["--enable-unsafe-webgpu"],
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.goto(process.argv[4]);
    const evidence = await page.evaluate(async repeats => {
      const { WebGPUUPDEBackend } = await import("/runner.mjs");
      const shader = await (await fetch("/kernel.wgsl")).text();
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("No actual WebGPU adapter");
      const info = adapter.info;
      const start = performance.now();
      const backend = await WebGPUUPDEBackend.create(shader);
      const setupMs = performance.now() - start;
      const period = Math.fround(2 * Math.PI);
      const interior = new Float32Array(new Uint32Array([
        new Uint32Array(new Float32Array([period]).buffer)[0] - 1,
      ]).buffer)[0];
      const initial = [0, -period, -2 * period, -0, period, interior, 0.25];
      const cases = [];
      for (const nSubsteps of [1, 3]) {
        const n = initial.length;
        const inputs = { phases: new Float32Array(initial),
          omegas: new Float32Array([-1e-15, 0, 0, 0, 0, 0, 2]),
          knm: new Float32Array(n*n), alpha: new Float32Array(n*n),
          zeta: 0, psi: 0, dt: 0.01, nSteps: 1, nSubsteps };
        const before = Array.from(new Uint32Array(inputs.phases.buffer));
        await backend.runEuler(inputs);
        const begin = performance.now();
        let output;
        for (let i = 0; i < repeats; i++) output = await backend.runEuler(inputs);
        cases.push({ nSubsteps, output: Array.from(output),
          negativeZero: Array.from(output, value => Object.is(value, -0)),
          inputBitsBefore: before,
          inputBitsAfter: Array.from(new Uint32Array(inputs.phases.buffer)),
          interior, msPerCall: (performance.now() - begin) / repeats });
      }
      const coupledInput = { phases: new Float32Array([0.1, 0.7]),
        omegas: new Float32Array([0.2, -0.3]),
        knm: new Float32Array([0, 0.8, 0.5, 0]),
        alpha: new Float32Array([0, 0.1, -0.2, 0]),
        zeta: 0.15, psi: 0.4, dt: 0.01, nSteps: 3, nSubsteps: 2 };
      const coupled = Array.from(await backend.runEuler(coupledInput));
      const refusals = [];
      for (const dt of [1e308, 1e-300, 0, -0.01]) {
        try { await backend.runEuler({ ...coupledInput, dt });
          throw new Error(`Invalid dt ${dt} was accepted`);
        } catch (error) {
          if (!(error instanceof RangeError)) throw error;
          refusals.push({ dt, error: String(error) });
        }
      }
      const divergentInput = { phases: new Float32Array([0]),
        omegas: new Float32Array([3e38]), knm: new Float32Array([0]),
        alpha: new Float32Array([0]), zeta: 3e38, psi: Math.PI / 2,
        dt: 0.01, nSteps: 1, nSubsteps: 1 };
      const divergentBits = Array.from(new Uint32Array(divergentInput.phases.buffer));
      let numericalRefusal;
      try { await backend.runEuler(divergentInput);
        throw new Error("Real float32 derivative overflow was accepted");
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        numericalRefusal = { error: String(error),
          inputBitsBefore: divergentBits,
          inputBitsAfter: Array.from(new Uint32Array(divergentInput.phases.buffer)) };
      }
      const recovered = Array.from(await backend.runEuler(coupledInput));
      backend.device.destroy();
      const deviceLost = await backend.device.lost;
      let deviceLossRefusal;
      try { await backend.runEuler(coupledInput);
        throw new Error("Destroyed WebGPU device was accepted");
      } catch (error) {
        if (!(error instanceof DOMException)) throw error;
        deviceLossRefusal = { reason: deviceLost.reason, name: error.name };
      }
      return { period, cases, coupled, recovered, refusals, numericalRefusal,
        deviceLossRefusal, setupMs,
        adapter: { vendor: info.vendor, architecture: info.architecture,
          device: info.device, description: info.description,
          isFallbackAdapter: info.isFallbackAdapter },
        repeats, scalarType: "f32", method: "euler",
        timingScope: "buffer-upload-submission-readback", hardwareSpeedupClaim: false };
    }, Number(process.argv[5]));
    if (errors.length) throw new Error(errors.join("\n"));
    console.log(JSON.stringify({ ...evidence, browserVersion: browser.version() }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def benchmark_webgpu_phase_wrapping(
    playwright_package: Path, browser_executable: Path, repeats: int = 3
) -> dict[str, object]:
    """Execute compiled WGSL through the public generated browser backend.

    Parameters
    ----------
    playwright_package : Path
        Installed Node ``playwright`` package directory, resolved by Node.
    browser_executable : Path
        Chromium executable supporting WebGPU on secure loopback.
    repeats : int, default 3
        Measured calls per boundary case, after one warm-up call.

    Returns
    -------
    dict[str, object]
        Actual browser outputs, refusal/recovery results, adapter identity,
        setup and readback timings, and generated-source SHA-256 digests.

    Raises
    ------
    ValueError
        If ``repeats`` is not a positive integer.
    subprocess.CalledProcessError
        If the real browser fails to compile or execute the package.
    RuntimeError
        If browser evidence is not a JSON object.

    Notes
    -----
    Only the owned browser, HTTP server, and temporary assets are closed.
    Shared-host timings and fallback adapters do not establish hardware speedup.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    node = shutil.which("node")
    if node is None:
        raise FileNotFoundError("An installed Node.js runtime is required")
    package = build_webgpu_upde_package()
    with tempfile.TemporaryDirectory(prefix="spo-webgpu-phase-") as temporary:
        directory = Path(temporary)
        (directory / "kernel.wgsl").write_text(package.wgsl, encoding="utf-8")
        (directory / "runner.mjs").write_text(package.javascript, encoding="utf-8")
        (directory / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>UPDE runtime</title>',
            encoding="utf-8",
        )
        driver = directory / "driver.cjs"
        driver.write_text(_DRIVER, encoding="utf-8")
        handler = functools.partial(SimpleHTTPRequestHandler, directory=temporary)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    node,
                    str(driver),
                    str(playwright_package.resolve()),
                    str(browser_executable.resolve()),
                    f"http://127.0.0.1:{server.server_port}/",
                    str(repeats),
                ],
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        decoded: object = json.loads(result.stdout)
        if not isinstance(decoded, dict):
            raise RuntimeError("Browser evidence must be a JSON object")
        evidence = cast("dict[str, object]", decoded)
    evidence["wgsl_sha256"] = sha256(package.wgsl.encode()).hexdigest()
    evidence["javascript_sha256"] = sha256(package.javascript.encode()).hexdigest()
    evidence["evidence_kind"] = "local_regression_non_isolated"
    return evidence


def main() -> int:
    """Run the real browser benchmark from explicit runtime paths.

    Returns
    -------
    int
        Zero after successful browser execution and JSON evidence output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playwright-package", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = benchmark_webgpu_phase_wrapping(
        args.playwright_package, args.browser_executable, args.repeats
    )
    text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
