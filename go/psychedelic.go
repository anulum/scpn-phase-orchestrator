// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Psychedelic observables (Go port)

// Package main builds ``libpsychedelic.so`` — a C-shared library
// exporting the circular-phase Shannon entropy kernel.
//
// Build with::
//
//	go build -buildmode=c-shared -o libpsychedelic.so psychedelic.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiPsyc = 2.0 * math.Pi

func entropyFromPhases(phases []float64, nBins int) float64 {
	t := len(phases)
	if t == 0 {
		return 0.0
	}
	counts := make([]int64, nBins)
	binWidth := twoPiPsyc / float64(nBins)
	for i := 0; i < t; i++ {
		v := math.Mod(phases[i], twoPiPsyc)
		if v < 0 {
			v += twoPiPsyc
		}
		bx := int(math.Floor(v / binWidth))
		if bx >= nBins {
			bx = nBins - 1
		}
		counts[bx]++
	}
	total := float64(t)
	h := 0.0
	for _, c := range counts {
		if c > 0 {
			p := float64(c) / total
			h -= p * math.Log(p)
		}
	}
	return h
}

//export EntropyFromPhases
func EntropyFromPhases(
	phasesPtr *C.double,
	t C.int,
	nBins C.int,
	outVal *C.double,
) C.int {
	tt := int(t)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt)
	*outVal = C.double(entropyFromPhases(phases, int(nBins)))
	return 0
}

func main() {}
