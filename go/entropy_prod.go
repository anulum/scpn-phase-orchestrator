// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Entropy production rate (Go port)

// Package main builds ``libentropy_prod.so`` — a C-shared library
// exporting the Kuramoto thermodynamic dissipation rate
// Σ (dθ/dt)² · dt.
//
// Build with::
//
//	go build -buildmode=c-shared -o libentropy_prod.so entropy_prod.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func entropyProductionRate(
	phases, omegas, knm []float64,
	alpha, dt float64,
	n int,
) float64 {
	if n == 0 || dt <= 0.0 {
		return 0.0
	}
	invN := alpha / float64(n)
	acc := 0.0
	for i := 0; i < n; i++ {
		s := 0.0
		offset := i * n
		for j := 0; j < n; j++ {
			s += knm[offset+j] * math.Sin(phases[j]-phases[i])
		}
		d := omegas[i] + invN*s
		acc += d * d
	}
	return acc * dt
}

//export EntropyProductionRate
//
// EntropyProductionRate writes a single f64 scalar into outVal.
func EntropyProductionRate(
	phasesPtr *C.double,
	omegasPtr *C.double,
	knmPtr *C.double,
	n C.int,
	alpha C.double,
	dt C.double,
	outVal *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	*outVal = C.double(
		entropyProductionRate(phases, omegas, knm, float64(alpha), float64(dt), nn),
	)
	return 0
}

func main() {}
