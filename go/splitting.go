// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Strang operator splitting (Go port)

// Package main builds ``libsplitting.so`` — Strang second-order
// operator splitting (A-B-A) for the Kuramoto ODE. Mirrors
// ``spo-engine/src/splitting.rs`` including the sincos expansion
// on the alpha-zero branch.
//
// Build with::
//
//	go build -buildmode=c-shared -o libsplitting.so splitting.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiSplit = 2.0 * math.Pi

func modTwoPi(x float64) float64 {
	v := math.Mod(x, twoPiSplit)
	if v < 0 {
		v += twoPiSplit
	}
	return v
}

func computeCouplingDeriv(
	theta, knm, alpha []float64,
	n int,
	zeta, psi float64,
	alphaZero bool,
	sinTh, cosTh, out []float64,
) {
	for i := 0; i < n; i++ {
		sinTh[i] = math.Sin(theta[i])
		cosTh[i] = math.Cos(theta[i])
	}
	var zsPsi, zcPsi float64
	if zeta != 0.0 {
		zsPsi = zeta * math.Sin(psi)
		zcPsi = zeta * math.Cos(psi)
	}
	for i := 0; i < n; i++ {
		offset := i * n
		ci := cosTh[i]
		si := sinTh[i]
		acc := 0.0
		if alphaZero {
			for j := 0; j < n; j++ {
				acc += knm[offset+j] * (sinTh[j]*ci - cosTh[j]*si)
			}
		} else {
			for j := 0; j < n; j++ {
				acc += knm[offset+j] *
					math.Sin(theta[j]-theta[i]-alpha[offset+j])
			}
		}
		out[i] = acc
		if zeta != 0.0 {
			out[i] += zsPsi*ci - zcPsi*si
		}
	}
}

func rk4Coupling(
	p, knm, alpha []float64,
	n int,
	zeta, psi, dt float64,
	alphaZero bool,
	k1, k2, k3, k4, tmp, sinTh, cosTh []float64,
) {
	computeCouplingDeriv(p, knm, alpha, n, zeta, psi, alphaZero,
		sinTh, cosTh, k1)
	for i := 0; i < n; i++ {
		tmp[i] = modTwoPi(p[i] + 0.5*dt*k1[i])
	}
	computeCouplingDeriv(tmp, knm, alpha, n, zeta, psi, alphaZero,
		sinTh, cosTh, k2)
	for i := 0; i < n; i++ {
		tmp[i] = modTwoPi(p[i] + 0.5*dt*k2[i])
	}
	computeCouplingDeriv(tmp, knm, alpha, n, zeta, psi, alphaZero,
		sinTh, cosTh, k3)
	for i := 0; i < n; i++ {
		tmp[i] = modTwoPi(p[i] + dt*k3[i])
	}
	computeCouplingDeriv(tmp, knm, alpha, n, zeta, psi, alphaZero,
		sinTh, cosTh, k4)
	dt6 := dt / 6.0
	for i := 0; i < n; i++ {
		p[i] = modTwoPi(p[i] + dt6*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]))
	}
}

func splittingRun(
	phases, omegas, knm, alpha []float64,
	n int,
	zeta, psi, dt float64,
	nSteps int,
	out []float64,
) {
	copy(out, phases)
	alphaZero := true
	for _, a := range alpha {
		if a != 0.0 {
			alphaZero = false
			break
		}
	}
	k1 := make([]float64, n)
	k2 := make([]float64, n)
	k3 := make([]float64, n)
	k4 := make([]float64, n)
	tmp := make([]float64, n)
	sinTh := make([]float64, n)
	cosTh := make([]float64, n)
	halfDt := 0.5 * dt
	for s := 0; s < nSteps; s++ {
		for i := 0; i < n; i++ {
			out[i] = modTwoPi(out[i] + halfDt*omegas[i])
		}
		rk4Coupling(out, knm, alpha, n, zeta, psi, dt, alphaZero,
			k1, k2, k3, k4, tmp, sinTh, cosTh)
		for i := 0; i < n; i++ {
			out[i] = modTwoPi(out[i] + halfDt*omegas[i])
		}
	}
}

//export SplittingRun
func SplittingRun(
	phasesPtr, omegasPtr, knmPtr, alphaPtr *C.double,
	n C.int,
	zeta, psi, dt C.double,
	nSteps C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	splittingRun(phases, omegas, knm, alpha, nn,
		float64(zeta), float64(psi), float64(dt),
		int(nSteps), out)
	return 0
}

func main() {}
