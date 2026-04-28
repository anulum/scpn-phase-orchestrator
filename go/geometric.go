// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Torus symplectic Euler (Go port)

// Package main builds ``libgeometric.so`` — torus-preserving
// geometric integrator (symplectic Euler on T^N). Mirrors
// ``spo-engine/src/geometric.rs`` bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libgeometric.so geometric.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiGeom = 2.0 * math.Pi

func torusRun(
	phases, omegas, knm, alpha []float64,
	n int,
	zeta, psi, dt float64,
	nSteps int,
	out []float64,
) {
	zRe := make([]float64, n)
	zIm := make([]float64, n)
	for i := 0; i < n; i++ {
		zRe[i] = math.Cos(phases[i])
		zIm[i] = math.Sin(phases[i])
	}

	alphaZero := true
	for _, a := range alpha {
		if a != 0.0 {
			alphaZero = false
			break
		}
	}
	var zsPsi, zcPsi float64
	if zeta != 0.0 {
		zsPsi = zeta * math.Sin(psi)
		zcPsi = zeta * math.Cos(psi)
	}

	nextRe := make([]float64, n)
	nextIm := make([]float64, n)

	for s := 0; s < nSteps; s++ {
		for i := 0; i < n; i++ {
			coupling := 0.0
			offset := i * n
			if alphaZero {
				for j := 0; j < n; j++ {
					coupling += knm[offset+j] *
						(zIm[j]*zRe[i] - zRe[j]*zIm[i])
				}
			} else {
				ti := math.Atan2(zIm[i], zRe[i])
				for j := 0; j < n; j++ {
					tj := math.Atan2(zIm[j], zRe[j])
					coupling += knm[offset+j] *
						math.Sin(tj-ti-alpha[offset+j])
				}
			}
			omegaEff := omegas[i] + coupling
			if zeta != 0.0 {
				omegaEff += zsPsi*zRe[i] - zcPsi*zIm[i]
			}
			angle := omegaEff * dt
			sinA := math.Sin(angle)
			cosA := math.Cos(angle)
			nr := zRe[i]*cosA - zIm[i]*sinA
			ni := zRe[i]*sinA + zIm[i]*cosA
			norm := math.Sqrt(nr*nr + ni*ni)
			if norm > 0.0 {
				nextRe[i] = nr / norm
				nextIm[i] = ni / norm
			} else {
				nextRe[i] = nr
				nextIm[i] = ni
			}
		}
		copy(zRe, nextRe)
		copy(zIm, nextIm)
	}

	for i := 0; i < n; i++ {
		v := math.Mod(math.Atan2(zIm[i], zRe[i]), twoPiGeom)
		if v < 0 {
			v += twoPiGeom
		}
		out[i] = v
	}
}

//export TorusRun
func TorusRun(
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
	torusRun(phases, omegas, knm, alpha, nn,
		float64(zeta), float64(psi), float64(dt),
		int(nSteps), out)
	return 0
}

func main() {}
