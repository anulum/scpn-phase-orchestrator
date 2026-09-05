// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — WASM playground simulation helpers

// Pure, DOM-free, WASM-free helpers for the Kuramoto playground. They mirror the
// mean-field order-parameter maths of `WasmEngine` so the page logic can be unit
// tested with `node --test` independently of the browser and the compiled WASM.

export const TWO_PI = 2.0 * Math.PI;

export const PARAM_LIMITS = Object.freeze({
  nMin: 2,
  nMax: 5000,
  couplingMin: 0,
  couplingMax: 50,
  dtMin: 1e-4,
  dtMax: 0.5,
  freqSpreadMin: 0,
  freqSpreadMax: 10,
});

export const SCENARIO_PRESETS = Object.freeze({
  weak_coupling: Object.freeze({
    label: "Weak coupling drift",
    description: "Low K keeps the oscillator cloud dispersed for comparison.",
    params: Object.freeze({ n: 48, coupling: 0.3, dt: 0.05, freqSpread: 1.0 }),
  }),
  critical_transition: Object.freeze({
    label: "Critical transition",
    description: "Near-threshold K shows the onset of partial synchronisation.",
    params: Object.freeze({ n: 64, coupling: 1.45, dt: 0.04, freqSpread: 1.2 }),
  }),
  strong_synchronisation: Object.freeze({
    label: "Strong synchronisation",
    description: "High K drives a rapid climb toward a locked mean field.",
    params: Object.freeze({ n: 64, coupling: 5.0, dt: 0.03, freqSpread: 0.6 }),
  }),
  wide_dispersion: Object.freeze({
    label: "Wide frequency dispersion",
    description: "Large frequency spread stresses coherence recovery.",
    params: Object.freeze({ n: 96, coupling: 2.4, dt: 0.035, freqSpread: 3.0 }),
  }),
});

/**
 * Validate playground parameters, returning a normalised copy.
 * @param {{n:number,coupling:number,dt:number,freqSpread:number}} params
 * @returns {{n:number,coupling:number,dt:number,freqSpread:number}}
 */
export function validateParams({ n, coupling, dt, freqSpread }) {
  if (!Number.isInteger(n) || n < PARAM_LIMITS.nMin || n > PARAM_LIMITS.nMax) {
    throw new RangeError(
      `n must be an integer in [${PARAM_LIMITS.nMin}, ${PARAM_LIMITS.nMax}]`,
    );
  }
  requireFiniteRange("coupling", coupling, PARAM_LIMITS.couplingMin, PARAM_LIMITS.couplingMax);
  requireFiniteRange("dt", dt, PARAM_LIMITS.dtMin, PARAM_LIMITS.dtMax);
  requireFiniteRange(
    "freqSpread",
    freqSpread,
    PARAM_LIMITS.freqSpreadMin,
    PARAM_LIMITS.freqSpreadMax,
  );
  return { n, coupling, dt, freqSpread };
}

/**
 * Return the deterministic scenario selector entries in display order.
 * @returns {Array<{id:string,label:string,description:string}>}
 */
export function scenarioOptions() {
  return Object.entries(SCENARIO_PRESETS).map(([id, preset]) => ({
    id,
    label: preset.label,
    description: preset.description,
  }));
}

/**
 * Return a defensive, validated parameter copy for a named scenario.
 * @param {string} id
 * @returns {{n:number,coupling:number,dt:number,freqSpread:number}}
 */
export function scenarioParams(id) {
  const preset = SCENARIO_PRESETS[id];
  if (preset === undefined) {
    throw new RangeError(`unknown playground scenario: ${id}`);
  }
  return validateParams({ ...preset.params });
}

function requireFiniteRange(name, value, min, max) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new RangeError(`${name} must be a finite number in [${min}, ${max}]`);
  }
}

/**
 * Build a deterministic, mean-centred linear spread of natural frequencies.
 * @param {number} n - oscillator count
 * @param {number} freqSpread - peak-to-peak spread in rad/s
 * @param {number} [base=1.0] - centre frequency in rad/s
 * @returns {Float64Array}
 */
export function buildOmegas(n, freqSpread, base = 1.0) {
  const omegas = new Float64Array(n);
  if (n === 1) {
    omegas[0] = base;
    return omegas;
  }
  const half = (n - 1) / 2;
  for (let i = 0; i < n; i += 1) {
    omegas[i] = base + freqSpread * ((i - half) / (n - 1));
  }
  return omegas;
}

/**
 * Evenly spread initial phases over the ring with a deterministic offset.
 * @param {number} n
 * @returns {Float64Array}
 */
export function spreadPhases(n) {
  const phases = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    phases[i] = (TWO_PI * i) / n;
  }
  return phases;
}

/**
 * Mean-field order parameter R = |mean(e^{i theta})| in [0, 1].
 * @param {ArrayLike<number>} phases
 * @returns {number}
 */
export function orderParameter(phases) {
  const n = phases.length;
  if (n === 0) {
    return 0;
  }
  let sumSin = 0;
  let sumCos = 0;
  for (let i = 0; i < n; i += 1) {
    sumSin += Math.sin(phases[i]);
    sumCos += Math.cos(phases[i]);
  }
  return Math.sqrt(sumSin * sumSin + sumCos * sumCos) / n;
}

/**
 * Mean-field phase psi = atan2(mean sin, mean cos) in (-pi, pi].
 * @param {ArrayLike<number>} phases
 * @returns {number}
 */
export function meanPhase(phases) {
  let sumSin = 0;
  let sumCos = 0;
  for (let i = 0; i < phases.length; i += 1) {
    sumSin += Math.sin(phases[i]);
    sumCos += Math.cos(phases[i]);
  }
  return Math.atan2(sumSin, sumCos);
}

/**
 * Wrap an angle to [0, 2*pi).
 * @param {number} theta
 * @returns {number}
 */
export function wrapPhase(theta) {
  return theta - Math.floor(theta / TWO_PI) * TWO_PI;
}

/**
 * Cartesian point on a circle for canvas rendering.
 * @param {number} theta
 * @param {number} radius
 * @param {number} cx
 * @param {number} cy
 * @returns {{x:number,y:number}}
 */
export function phasePoint(theta, radius, cx, cy) {
  return { x: cx + radius * Math.cos(theta), y: cy - radius * Math.sin(theta) };
}

/**
 * Operator-style coherence label for a given order parameter.
 * @param {number} r
 * @returns {"incoherent"|"partial"|"synchronised"}
 */
export function coherenceLabel(r) {
  if (r < 0.3) {
    return "incoherent";
  }
  if (r < 0.8) {
    return "partial";
  }
  return "synchronised";
}
