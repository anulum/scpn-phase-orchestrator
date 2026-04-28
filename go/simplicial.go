// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Simplicial Kuramoto (Go port)

// Package main builds ``libsimplicial.so`` — pairwise + 3-body
// all-to-all Kuramoto stepper using the Gambuzza et al. 2023
// closed-form ``Σ sin(θ_j + θ_k − 2θ_i) = 2·S_i·C_i`` identity.
//
// Build with::
//
//	go build -buildmode=c-shared -o libsimplicial.so simplicial.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiSimp = 2.0 * math.Pi

func simplicialStep(
	theta, omegas, knm, alpha []float64,
	n int,
	zeta, psi, sigma2 float64,
	alphaZero bool,
	sinTh, cosTh, deriv []float64,
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
	var gs, gc float64
	use3Body := sigma2 != 0.0 && n >= 3
	if use3Body {
		for i := 0; i < n; i++ {
			gs += sinTh[i]
			gc += cosTh[i]
		}
	}
	var invN2 float64
	if n > 0 {
		invN2 = sigma2 / (float64(n) * float64(n))
	}

	for i := 0; i < n; i++ {
		offset := i * n
		ci := cosTh[i]
		si := sinTh[i]
		pw := 0.0
		if alphaZero {
			for j := 0; j < n; j++ {
				pw += knm[offset+j] * (sinTh[j]*ci - cosTh[j]*si)
			}
		} else {
			for j := 0; j < n; j++ {
				pw += knm[offset+j] *
					math.Sin(theta[j]-theta[i]-alpha[offset+j])
			}
		}
		deriv[i] = omegas[i] + pw
		if use3Body {
			deriv[i] += 2.0 * (gs*ci - gc*si) * (gc*ci + gs*si) * invN2
		}
		if zeta != 0.0 {
			deriv[i] += zsPsi*ci - zcPsi*si
		}
	}
}

func simplicialRun(
	phases, omegas, knm, alpha []float64,
	n int,
	zeta, psi, sigma2, dt float64,
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
	sinTh := make([]float64, n)
	cosTh := make([]float64, n)
	deriv := make([]float64, n)
	for s := 0; s < nSteps; s++ {
		simplicialStep(out, omegas, knm, alpha, n,
			zeta, psi, sigma2, alphaZero,
			sinTh, cosTh, deriv)
		for i := 0; i < n; i++ {
			v := math.Mod(out[i]+dt*deriv[i], twoPiSimp)
			if v < 0 {
				v += twoPiSimp
			}
			out[i] = v
		}
	}
}

//export SimplicialRun
func SimplicialRun(
	phasesPtr, omegasPtr, knmPtr, alphaPtr *C.double,
	n C.int,
	zeta, psi, sigma2, dt C.double,
	nSteps C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	simplicialRun(phases, omegas, knm, alpha, nn,
		float64(zeta), float64(psi), float64(sigma2), float64(dt),
		int(nSteps), out)
	return 0
}

func main() {}
