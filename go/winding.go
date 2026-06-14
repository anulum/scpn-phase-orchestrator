// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase winding numbers (Go port)

// Package main builds “libwinding.so“ — a C-shared library
// exporting the cumulative winding number per oscillator.
//
// Build with::
//
//	go build -buildmode=c-shared -o libwinding.so winding.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiWind = 2.0 * math.Pi

// wrap maps an increment into the half-open interval (-π, π] so an exact
// forward half-turn (+π) counts forward rather than aliasing to -π.
func wrap(delta float64) float64 {
	r := math.Mod(math.Pi-delta, twoPiWind)
	if r < 0.0 {
		r += twoPiWind
	}
	return math.Pi - r
}

func windingNumbers(phases []float64, t, n int, out []int64) {
	if t < 2 {
		for i := 0; i < n; i++ {
			out[i] = 0
		}
		return
	}
	cumulative := make([]float64, n)
	for step := 1; step < t; step++ {
		baseNow := step * n
		basePrev := (step - 1) * n
		for i := 0; i < n; i++ {
			delta := phases[baseNow+i] - phases[basePrev+i]
			cumulative[i] += wrap(delta)
		}
	}
	for i := 0; i < n; i++ {
		out[i] = int64(math.Floor(cumulative[i] / twoPiWind))
	}
}

// WindingNumbers writes n int64 winding numbers into outPtr.
// phasesPtr is the flat row-major (T, N) matrix.
//
//export WindingNumbers
func WindingNumbers(
	phasesPtr *C.double,
	t C.int,
	n C.int,
	outPtr *C.longlong,
) C.int {
	tt := int(t)
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt*nn)
	out := unsafe.Slice((*int64)(unsafe.Pointer(outPtr)), nn)
	windingNumbers(phases, tt, nn, out)
	return 0
}

func main() {}
