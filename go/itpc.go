// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Inter-Trial Phase Coherence (Go port)

// Package main builds ``libitpc.so`` — a C-shared library exporting
// the Lachaux 1999 ITPC kernel plus its stimulus-pause persistence
// variant. Matches the NumPy / Rust / Julia / Mojo references
// bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libitpc.so itpc.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func computeITPC(phases []float64, nTrials, nTp int, out []float64) {
	if nTrials == 0 {
		return
	}
	invN := 1.0 / float64(nTrials)
	for t := 0; t < nTp; t++ {
		sr, si := 0.0, 0.0
		for k := 0; k < nTrials; k++ {
			th := phases[k*nTp+t]
			sr += math.Cos(th)
			si += math.Sin(th)
		}
		sr *= invN
		si *= invN
		out[t] = math.Sqrt(sr*sr + si*si)
	}
}

//export ComputeITPC
//
// ComputeITPC writes n_timepoints f64 entries into outPtr.
// phasesPtr is the flat row-major (n_trials, n_timepoints) matrix.
func ComputeITPC(
	phasesPtr *C.double,
	nTrials C.int,
	nTp C.int,
	outPtr *C.double,
) C.int {
	nT := int(nTrials)
	nP := int(nTp)
	if nT == 0 || nP == 0 {
		return 0
	}
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nT*nP)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nP)
	computeITPC(phases, nT, nP, out)
	return 0
}

//export ITPCPersistence
//
// ITPCPersistence writes a single f64 mean into outVal.
// pauseIndicesPtr holds nIdx int64 zero-based indices.
func ITPCPersistence(
	phasesPtr *C.double,
	nTrials C.int,
	nTp C.int,
	pauseIndicesPtr *C.longlong,
	nIdx C.int,
	outVal *C.double,
) C.int {
	nT := int(nTrials)
	nP := int(nTp)
	nI := int(nIdx)
	*outVal = 0.0
	if nI == 0 || nT == 0 || nP == 0 {
		return 0
	}
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nT*nP)
	pause := unsafe.Slice(
		(*int64)(unsafe.Pointer(pauseIndicesPtr)), nI,
	)
	itpc := make([]float64, nP)
	computeITPC(phases, nT, nP, itpc)
	acc := 0.0
	count := 0
	for _, idx := range pause {
		if idx >= 0 && int(idx) < nP {
			acc += itpc[int(idx)]
			count++
		}
	}
	if count > 0 {
		*outVal = C.double(acc / float64(count))
	}
	return 0
}

func main() {}
