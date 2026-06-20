// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — WASM playground simulation helper tests

// Run with: node --test spo-kernel/crates/spo-wasm/example/

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  buildOmegas,
  coherenceLabel,
  meanPhase,
  orderParameter,
  PARAM_LIMITS,
  phasePoint,
  spreadPhases,
  TWO_PI,
  validateParams,
  wrapPhase,
} from "./simulation.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = resolve(HERE, "../../../../wasm-pkg");

const VALID = { n: 32, coupling: 2.0, dt: 0.05, freqSpread: 0.4 };

test("validateParams accepts valid parameters", () => {
  assert.deepEqual(validateParams(VALID), VALID);
});

test("validateParams rejects out-of-range n", () => {
  assert.throws(() => validateParams({ ...VALID, n: 1 }), RangeError);
  assert.throws(() => validateParams({ ...VALID, n: PARAM_LIMITS.nMax + 1 }), RangeError);
  assert.throws(() => validateParams({ ...VALID, n: 4.5 }), RangeError);
});

test("validateParams rejects out-of-range continuous parameters", () => {
  assert.throws(() => validateParams({ ...VALID, coupling: -1 }), RangeError);
  assert.throws(() => validateParams({ ...VALID, dt: 0 }), RangeError);
  assert.throws(() => validateParams({ ...VALID, dt: Number.POSITIVE_INFINITY }), RangeError);
  assert.throws(() => validateParams({ ...VALID, freqSpread: -0.1 }), RangeError);
});

test("buildOmegas is mean-centred and the right length", () => {
  const omegas = buildOmegas(64, 0.6, 1.0);
  assert.equal(omegas.length, 64);
  const mean = omegas.reduce((a, b) => a + b, 0) / omegas.length;
  assert.ok(Math.abs(mean - 1.0) < 1e-12);
  // Symmetric spread: first and last are mirror images about the base.
  assert.ok(Math.abs((omegas[0] - 1.0) + (omegas[63] - 1.0)) < 1e-12);
});

test("buildOmegas handles the singleton case", () => {
  const omegas = buildOmegas(1, 0.6, 2.0);
  assert.deepEqual(Array.from(omegas), [2.0]);
});

test("spreadPhases distributes evenly over the ring", () => {
  const phases = spreadPhases(4);
  assert.equal(phases.length, 4);
  assert.ok(Math.abs(phases[1] - TWO_PI / 4) < 1e-12);
});

test("orderParameter is 1 for identical phases and ~0 for an even spread", () => {
  assert.ok(Math.abs(orderParameter(new Float64Array(16).fill(0.7)) - 1.0) < 1e-12);
  assert.ok(orderParameter(spreadPhases(64)) < 1e-10);
  assert.equal(orderParameter(new Float64Array(0)), 0);
});

test("meanPhase recovers a known mean direction", () => {
  assert.ok(Math.abs(meanPhase(new Float64Array([0.3, 0.3, 0.3])) - 0.3) < 1e-12);
});

test("wrapPhase maps any angle into [0, 2*pi)", () => {
  assert.ok(Math.abs(wrapPhase(-0.1) - (TWO_PI - 0.1)) < 1e-12);
  assert.ok(Math.abs(wrapPhase(TWO_PI + 0.2) - 0.2) < 1e-12);
});

test("phasePoint lands on the circle", () => {
  const p = phasePoint(1.1, 50, 100, 100);
  assert.ok(Math.abs(Math.hypot(p.x - 100, p.y - 100) - 50) < 1e-9);
});

test("coherenceLabel partitions the order parameter", () => {
  assert.equal(coherenceLabel(0.1), "incoherent");
  assert.equal(coherenceLabel(0.5), "partial");
  assert.equal(coherenceLabel(0.95), "synchronised");
});

test("WASM engine synchronises as coupling grows", async (t) => {
  const glue = resolve(WASM_DIR, "spo_wasm.js");
  const binary = resolve(WASM_DIR, "spo_wasm_bg.wasm");
  if (!existsSync(glue) || !existsSync(binary)) {
    t.skip("wasm-pkg not built; run `wasm-pack build --target web` first");
    return;
  }
  const { default: init, WasmEngine } = await import(glue);
  await init({ module_or_path: await readFile(binary) });

  const n = 24;
  const omegas = buildOmegas(n, 0.3, 1.0);
  const run = (coupling) => {
    const engine = new WasmEngine(n);
    engine.set_phases(spreadPhases(n));
    let r = 0;
    for (let step = 0; step < 400; step += 1) {
      r = engine.step(omegas, coupling, 0.05);
    }
    // `step` reports R of the current phases before advancing, so the pure
    // helper on a snapshot of those phases must equal the next step's return.
    // `get_phases` is a view into WASM memory, so copy it before stepping.
    const snapshot = Float64Array.from(engine.get_phases());
    const nextR = engine.step(omegas, coupling, 0.05);
    assert.ok(Math.abs(orderParameter(snapshot) - nextR) < 1e-9);
    return r;
  };

  const weak = run(0.0);
  const strong = run(5.0);
  assert.ok(strong > weak);
  assert.ok(strong > 0.9, `expected strong coupling to synchronise, got R=${strong}`);
});
