// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hodge decomposition (Go port)

// Package main builds ``libhodge.so`` — Hodge decomposition of a
// coupling matrix into gradient / curl / harmonic components per
// oscillator.
//
// Build with::
//
//	go build -buildmode=c-shared -o libhodge.so hodge.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func hodgeDecomposition(
	knm, phases []float64, n int,
	gradient, curl, harmonic []float64,
) {
	for i := 0; i < n; i++ {
		g, c, t := 0.0, 0.0, 0.0
		thetaI := phases[i]
		baseI := i * n
		for j := 0; j < n; j++ {
			kij := knm[baseI+j]
			kji := knm[j*n+i]
			cd := math.Cos(phases[j] - thetaI)
			sym := 0.5 * (kij + kji)
			anti := 0.5 * (kij - kji)
			g += sym * cd
			c += anti * cd
			t += kij * cd
		}
		gradient[i] = g
		curl[i] = c
		harmonic[i] = t - g - c
	}
}

//export HodgeDecomposition
func HodgeDecomposition(
	knmPtr *C.double, phasesPtr *C.double, n C.int,
	gPtr, cPtr, hPtr *C.double,
) C.int {
	nn := int(n)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	gradient := unsafe.Slice((*float64)(unsafe.Pointer(gPtr)), nn)
	curl := unsafe.Slice((*float64)(unsafe.Pointer(cPtr)), nn)
	harmonic := unsafe.Slice((*float64)(unsafe.Pointer(hPtr)), nn)
	hodgeDecomposition(knm, phases, nn, gradient, curl, harmonic)
	return 0
}

func main() {}
