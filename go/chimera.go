// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Chimera local order-parameter (Go port)

// Package main builds “libchimera.so“ — a C-shared library
// exporting the Kuramoto local order parameter per oscillator. The
// coherent / incoherent partition stays Python-side.
//
// Build with::
//
//	go build -buildmode=c-shared -o libchimera.so chimera.go
package main

// #include <stddef.h>
import "C"

import (
	"math"
	"unsafe"
)

func localOrderParameter(phases, knm []float64, n int, out []float64) {
	for i := 0; i < n; i++ {
		sr, si := 0.0, 0.0
		cnt := 0
		base := i * n
		for j := 0; j < n; j++ {
			if j != i && knm[base+j] > 0.0 {
				sr += math.Cos(phases[j])
				si += math.Sin(phases[j])
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

// LocalOrderParameter retains the valid legacy ABI. Callers must supply readable
// n and n*n double buffers and writable n doubles. Prefer the extent-aware V2.
//
//export LocalOrderParameter
func LocalOrderParameter(
	phasesPtr *C.double,
	knmPtr *C.double,
	n C.int,
	outPtr *C.double,
) C.int {
	if n < 0 {
		return 1
	}
	nn := uint64(n)
	return LocalOrderParameterV2(phasesPtr, C.size_t(nn), knmPtr, C.size_t(nn*nn),
		n, outPtr, C.size_t(nn))
}

// LocalOrderParameterV2 admits exact buffer extents before unsafe slice creation.
// Return 1 means invalid dimensions/pointers; 2 means nonfinite input or nonzero
// diagonal. Errors leave output unchanged. Pointer storage must really match the
// declared extents; metadata cannot prove C pointer allocation or liveness.
//
//export LocalOrderParameterV2
func LocalOrderParameterV2(
	phasesPtr *C.double, phasesLen C.size_t,
	knmPtr *C.double, knmLen C.size_t,
	n C.int,
	outPtr *C.double, outLen C.size_t,
) C.int {
	if n < 0 {
		return 1
	}
	nn := uint64(n)
	maxElements := uint64(int(^uint(0)>>1)) / uint64(unsafe.Sizeof(float64(0)))
	if nn > maxElements || (nn != 0 && nn > maxElements/nn) {
		return 1
	}
	if uint64(phasesLen) != nn || uint64(knmLen) != nn*nn || uint64(outLen) != nn {
		return 1
	}
	if nn == 0 {
		return 0
	}
	if phasesPtr == nil || knmPtr == nil || outPtr == nil {
		return 1
	}
	count := int(nn)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), count)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), count*count)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), count)
	for _, value := range phases {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 2
		}
	}
	for _, value := range knm {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 2
		}
	}
	for i := 0; i < count; i++ {
		if math.Abs(knm[i*count+i]) > 1e-15 {
			return 2
		}
	}
	localOrderParameter(phases, knm, count, out)
	return 0
}

func main() {}
