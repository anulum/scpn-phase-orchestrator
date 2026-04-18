// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Chimera local order-parameter (Go port)

// Package main builds ``libchimera.so`` — a C-shared library
// exporting the Kuramoto local order parameter per oscillator. The
// coherent / incoherent partition stays Python-side.
//
// Build with::
//
//	go build -buildmode=c-shared -o libchimera.so chimera.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func localOrderParameter(phases, knm []float64, n int, out []float64) {
	for i := 0; i < n; i++ {
		sr, si := 0.0, 0.0
		cnt := 0
		thetaI := phases[i]
		base := i * n
		for j := 0; j < n; j++ {
			if knm[base+j] > 0.0 {
				delta := phases[j] - thetaI
				sr += math.Cos(delta)
				si += math.Sin(delta)
				cnt++
			}
		}
		if cnt == 0 {
			out[i] = 0.0
			continue
		}
		inv := 1.0 / float64(cnt)
		sr *= inv
		si *= inv
		out[i] = math.Sqrt(sr*sr + si*si)
	}
}

//export LocalOrderParameter
//
// LocalOrderParameter writes n f64 R_i values into outPtr.
func LocalOrderParameter(
	phasesPtr *C.double,
	knmPtr *C.double,
	n C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	localOrderParameter(phases, knm, nn, out)
	return 0
}

func main() {}
