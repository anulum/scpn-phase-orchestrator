// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — UPDE batched integrator (Go port)

// Package main builds ``libupde_engine.so`` — a C-shared library
// exporting the batched Kuramoto / Sakaguchi UPDE integrator. Three
// integrators match the Rust reference exactly: Euler, RK4, and
// Dormand-Prince RK45 with adaptive step-size control.
//
// Build with::
//
//	go build -buildmode=c-shared -o libupde_engine.so upde_engine.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiUPDE = 2.0 * math.Pi

// Dormand-Prince tableau (matches spo-engine/src/dp_tableau.rs).
const (
	dpA21 = 1.0 / 5.0
	dpA31 = 3.0 / 40.0
	dpA32 = 9.0 / 40.0
	dpA41 = 44.0 / 45.0
	dpA42 = -56.0 / 15.0
	dpA43 = 32.0 / 9.0
	dpA51 = 19372.0 / 6561.0
	dpA52 = -25360.0 / 2187.0
	dpA53 = 64448.0 / 6561.0
	dpA54 = -212.0 / 729.0
	dpA61 = 9017.0 / 3168.0
	dpA62 = -355.0 / 33.0
	dpA63 = 46732.0 / 5247.0
	dpA64 = 49.0 / 176.0
	dpA65 = -5103.0 / 18656.0
	dpB5_0 = 35.0 / 384.0
	dpB5_2 = 500.0 / 1113.0
	dpB5_3 = 125.0 / 192.0
	dpB5_4 = -2187.0 / 6784.0
	dpB5_5 = 11.0 / 84.0
	dpB4_0 = 5179.0 / 57600.0
	dpB4_2 = 7571.0 / 16695.0
	dpB4_3 = 393.0 / 640.0
	dpB4_4 = -92097.0 / 339200.0
	dpB4_5 = 187.0 / 2100.0
	dpB4_6 = 1.0 / 40.0
)

func computeDerivative(
	out, theta, omegas, knm, alpha []float64,
	zeta, psi float64,
	n int,
) {
	for i := 0; i < n; i++ {
		s := 0.0
		offset := i * n
		for j := 0; j < n; j++ {
			s += knm[offset+j] * math.Sin(theta[j]-theta[i]-alpha[offset+j])
		}
		driving := 0.0
		if zeta != 0.0 {
			driving = zeta * math.Sin(psi-theta[i])
		}
		out[i] = omegas[i] + s + driving
	}
}

func eulerSubstep(
	phases, omegas, knm, alpha, buf []float64,
	zeta, psi, dt float64, n int,
) {
	computeDerivative(buf, phases, omegas, knm, alpha, zeta, psi, n)
	for i := 0; i < n; i++ {
		phases[i] += dt * buf[i]
	}
}

func rk4Substep(
	phases, omegas, knm, alpha []float64,
	k1, k2, k3, k4, tmp []float64,
	zeta, psi, dt float64, n int,
) {
	computeDerivative(k1, phases, omegas, knm, alpha, zeta, psi, n)
	for i := 0; i < n; i++ {
		tmp[i] = phases[i] + 0.5*dt*k1[i]
	}
	computeDerivative(k2, tmp, omegas, knm, alpha, zeta, psi, n)
	for i := 0; i < n; i++ {
		tmp[i] = phases[i] + 0.5*dt*k2[i]
	}
	computeDerivative(k3, tmp, omegas, knm, alpha, zeta, psi, n)
	for i := 0; i < n; i++ {
		tmp[i] = phases[i] + dt*k3[i]
	}
	computeDerivative(k4, tmp, omegas, knm, alpha, zeta, psi, n)
	dt6 := dt / 6.0
	for i := 0; i < n; i++ {
		phases[i] += dt6 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
	}
}

func rk45Step(
	phases, omegas, knm, alpha []float64,
	k1, k2, k3, k4, k5, k6, k7, y5, tmp []float64,
	zeta, psi, atol, rtol, dtConfig, lastDt float64,
	n int,
) float64 {
	dt := lastDt
	for attempt := 0; attempt < 4; attempt++ {
		computeDerivative(k1, phases, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			tmp[i] = phases[i] + dt*dpA21*k1[i]
		}
		computeDerivative(k2, tmp, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			tmp[i] = phases[i] + dt*(dpA31*k1[i]+dpA32*k2[i])
		}
		computeDerivative(k3, tmp, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			tmp[i] = phases[i] + dt*(dpA41*k1[i]+dpA42*k2[i]+dpA43*k3[i])
		}
		computeDerivative(k4, tmp, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			tmp[i] = phases[i] + dt*(
				dpA51*k1[i]+dpA52*k2[i]+dpA53*k3[i]+dpA54*k4[i])
		}
		computeDerivative(k5, tmp, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			tmp[i] = phases[i] + dt*(
				dpA61*k1[i]+dpA62*k2[i]+dpA63*k3[i]+dpA64*k4[i]+dpA65*k5[i])
		}
		computeDerivative(k6, tmp, omegas, knm, alpha, zeta, psi, n)
		for i := 0; i < n; i++ {
			y5[i] = phases[i] + dt*(
				dpB5_0*k1[i]+dpB5_2*k3[i]+dpB5_3*k4[i]+dpB5_4*k5[i]+dpB5_5*k6[i])
		}
		computeDerivative(k7, y5, omegas, knm, alpha, zeta, psi, n)
		errNorm := 0.0
		for i := 0; i < n; i++ {
			y4 := phases[i] + dt*(
				dpB4_0*k1[i]+dpB4_2*k3[i]+dpB4_3*k4[i]+
					dpB4_4*k5[i]+dpB4_5*k6[i]+dpB4_6*k7[i])
			errI := math.Abs(y5[i] - y4)
			scale := atol + rtol*math.Max(math.Abs(phases[i]), math.Abs(y5[i]))
			ratio := errI / scale
			if ratio > errNorm {
				errNorm = ratio
			}
		}
		if errNorm <= 1.0 {
			factor := 5.0
			if errNorm > 0.0 {
				factor = math.Min(0.9*math.Pow(errNorm, -0.2), 5.0)
			}
			newLast := math.Min(dt*factor, dtConfig*10.0)
			copy(phases[:n], y5[:n])
			return newLast
		}
		factor := math.Max(0.9*math.Pow(errNorm, -0.25), 0.2)
		dt *= factor
	}
	copy(phases[:n], y5[:n])
	return dt
}

func upderRun(
	phases []float64,
	omegas, knm, alpha []float64,
	zeta, psi, dt float64,
	nSteps, method int,
	nSubsteps int,
	atol, rtol float64,
	n int,
) {
	k1 := make([]float64, n)
	k2 := make([]float64, n)
	k3 := make([]float64, n)
	k4 := make([]float64, n)
	k5 := make([]float64, n)
	k6 := make([]float64, n)
	k7 := make([]float64, n)
	y5 := make([]float64, n)
	tmp := make([]float64, n)
	lastDt := dt
	subDt := dt / float64(nSubsteps)

	for step := 0; step < nSteps; step++ {
		switch method {
		case 2: // rk45
			lastDt = rk45Step(
				phases, omegas, knm, alpha,
				k1, k2, k3, k4, k5, k6, k7, y5, tmp,
				zeta, psi, atol, rtol, dt, lastDt, n,
			)
		case 1: // rk4
			for s := 0; s < nSubsteps; s++ {
				rk4Substep(
					phases, omegas, knm, alpha,
					k1, k2, k3, k4, tmp,
					zeta, psi, subDt, n,
				)
			}
		default: // euler
			for s := 0; s < nSubsteps; s++ {
				eulerSubstep(
					phases, omegas, knm, alpha, k1,
					zeta, psi, subDt, n,
				)
			}
		}
		for i := 0; i < n; i++ {
			phases[i] = math.Mod(phases[i], twoPiUPDE)
			if phases[i] < 0.0 {
				phases[i] += twoPiUPDE
			}
		}
	}
}

//export UPDERun
//
// UPDERun integrates n_steps of the Kuramoto UPDE starting from the
// phases slice (modified in place — caller copies in first).
// ``method`` is 0 = Euler, 1 = RK4, 2 = RK45.
// Returns 0 on success.
func UPDERun(
	phasesPtr *C.double,
	omegasPtr *C.double,
	knmPtr *C.double,
	alphaPtr *C.double,
	n C.int,
	zeta C.double,
	psi C.double,
	dt C.double,
	nSteps C.int,
	method C.int,
	nSubsteps C.int,
	atol C.double,
	rtol C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	upderRun(
		phases, omegas, knm, alpha,
		float64(zeta), float64(psi), float64(dt),
		int(nSteps), int(method),
		int(nSubsteps),
		float64(atol), float64(rtol),
		nn,
	)
	return 0
}

func main() {}
