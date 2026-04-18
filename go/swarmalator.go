// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Swarmalator stepper (Go port)

// Package main builds ``libswarmalator.so`` — a single
// O(N²·d) swarmalator step in position + phase space.
//
// Build with::
//
//	go build -buildmode=c-shared -o libswarmalator.so swarmalator.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiSwarm = 2.0 * math.Pi

func swarmalatorStep(
	pos, phases, omegas []float64,
	n, dim int,
	a, b, j, k, dt float64,
	newPos, newPhases []float64,
) {
	copy(newPos, pos)
	invN := 1.0 / float64(n)
	for i := 0; i < n; i++ {
		vel := make([]float64, dim)
		phaseAcc := 0.0
		baseI := i * dim
		thetaI := phases[i]
		for m := 0; m < n; m++ {
			baseM := m * dim
			s := 0.0
			for d := 0; d < dim; d++ {
				delta := pos[baseM+d] - pos[baseI+d]
				s += delta * delta
			}
			dist := math.Sqrt(s + 1e-6)
			cosD := math.Cos(phases[m] - thetaI)
			sinD := math.Sin(phases[m] - thetaI)
			attract := (a + j*cosD) / dist
			// Rust canonical: b / (dist * d2 + eps).
			repulse := b / (dist*s + 1e-6)
			factor := attract - repulse
			for d := 0; d < dim; d++ {
				delta := pos[baseM+d] - pos[baseI+d]
				vel[d] += delta * factor
			}
			phaseAcc += sinD / dist
		}
		for d := 0; d < dim; d++ {
			newPos[baseI+d] = pos[baseI+d] + dt*vel[d]*invN
		}
		dth := omegas[i] + k*phaseAcc*invN
		v := math.Mod(thetaI+dt*dth, twoPiSwarm)
		if v < 0 {
			v += twoPiSwarm
		}
		newPhases[i] = v
	}
}

//export SwarmalatorStep
func SwarmalatorStep(
	posPtr *C.double,
	phasesPtr *C.double,
	omegasPtr *C.double,
	n C.int,
	dim C.int,
	a C.double, b C.double, j C.double, k C.double, dt C.double,
	newPosPtr *C.double,
	newPhasesPtr *C.double,
) C.int {
	nn := int(n)
	dd := int(dim)
	pos := unsafe.Slice((*float64)(unsafe.Pointer(posPtr)), nn*dd)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	newPos := unsafe.Slice((*float64)(unsafe.Pointer(newPosPtr)), nn*dd)
	newPhases := unsafe.Slice((*float64)(unsafe.Pointer(newPhasesPtr)), nn)
	swarmalatorStep(
		pos, phases, omegas, nn, dd,
		float64(a), float64(b), float64(j), float64(k), float64(dt),
		newPos, newPhases,
	)
	return 0
}

func main() {}
