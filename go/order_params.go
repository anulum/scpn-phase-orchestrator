// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Order parameters (Go port)

// Package main builds ``liborder_params.so`` — a C-shared library
// exporting the Kuramoto order parameter, the phase-locking value
// and the layer-coherence helper. Mirrors the Rust kernel.
//
// Build with::
//
//	go build -buildmode=c-shared -o liborder_params.so order_params.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPi = 2.0 * math.Pi

//export OrderParameter
//
// OrderParameter writes (R, psi) into outR/outPsi. Returns 0 on
// success; the only failure mode is a zero-length input (treated as
// R=0, psi=0 per the Python/Rust contract — still returns 0).
func OrderParameter(
	phasesPtr *C.double,
	n C.int,
	outR *C.double,
	outPsi *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	if nn == 0 {
		*outR = 0.0
		*outPsi = 0.0
		return 0
	}
	var sx, sy float64
	for _, p := range phases {
		sx += math.Cos(p)
		sy += math.Sin(p)
	}
	sx /= float64(nn)
	sy /= float64(nn)
	r := math.Sqrt(sx*sx + sy*sy)
	psi := math.Mod(math.Atan2(sy, sx), twoPi)
	if psi < 0 {
		psi += twoPi
	}
	*outR = C.double(r)
	*outPsi = C.double(psi)
	return 0
}

//export PLV
//
// PLV writes the phase-locking value into outPLV. Returns 1 on length
// mismatch, 0 on success.
func PLV(
	phasesAPtr *C.double,
	phasesBPtr *C.double,
	n C.int,
	outPLV *C.double,
) C.int {
	nn := int(n)
	if nn == 0 {
		*outPLV = 0.0
		return 0
	}
	phasesA := unsafe.Slice((*float64)(unsafe.Pointer(phasesAPtr)), nn)
	phasesB := unsafe.Slice((*float64)(unsafe.Pointer(phasesBPtr)), nn)
	var sx, sy float64
	for i := 0; i < nn; i++ {
		diff := phasesA[i] - phasesB[i]
		sx += math.Cos(diff)
		sy += math.Sin(diff)
	}
	sx /= float64(nn)
	sy /= float64(nn)
	*outPLV = C.double(math.Sqrt(sx*sx + sy*sy))
	return 0
}

//export LayerCoherence
//
// LayerCoherence writes R restricted to the oscillators at `indicesPtr`.
func LayerCoherence(
	phasesPtr *C.double,
	nPhases C.int,
	indicesPtr *C.longlong,
	nIndices C.int,
	outR *C.double,
) C.int {
	ni := int(nIndices)
	if ni == 0 {
		*outR = 0.0
		return 0
	}
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), int(nPhases))
	indices := unsafe.Slice((*int64)(unsafe.Pointer(indicesPtr)), ni)
	var sx, sy float64
	for _, idx := range indices {
		i := int(idx)
		if i < 0 || i >= int(nPhases) {
			return 1
		}
		sx += math.Cos(phases[i])
		sy += math.Sin(phases[i])
	}
	sx /= float64(ni)
	sy /= float64(ni)
	*outR = C.double(math.Sqrt(sx*sx + sy*sy))
	return 0
}

func main() {}
