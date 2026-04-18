// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Envelope kernels (Go port)

// Package main builds ``libenvelope.so`` — sliding-window RMS and
// modulation-depth kernels.
//
// Build with::
//
//	go build -buildmode=c-shared -o libenvelope.so envelope.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func extractEnvelope(amps []float64, window int, out []float64) {
	t := len(amps)
	if t == 0 {
		return
	}
	cs := make([]float64, t+1)
	for i := 0; i < t; i++ {
		cs[i+1] = cs[i] + amps[i]*amps[i]
	}
	nValid := t - window + 1
	if nValid <= 0 {
		v := math.Sqrt(cs[t] / float64(t))
		for i := range out {
			out[i] = v
		}
		return
	}
	for i := 0; i < nValid; i++ {
		out[window-1+i] = math.Sqrt((cs[i+window] - cs[i]) / float64(window))
	}
	first := out[window-1]
	for i := 0; i < window-1; i++ {
		out[i] = first
	}
}

func envelopeModulationDepth(envelope []float64) float64 {
	t := len(envelope)
	if t == 0 {
		return 0.0
	}
	vmax := envelope[0]
	vmin := envelope[0]
	for i := 1; i < t; i++ {
		v := envelope[i]
		if v > vmax {
			vmax = v
		}
		if v < vmin {
			vmin = v
		}
	}
	denom := vmax + vmin
	if denom <= 0.0 {
		return 0.0
	}
	return (vmax - vmin) / denom
}

//export ExtractEnvelope
func ExtractEnvelope(
	ampsPtr *C.double,
	t C.int,
	window C.int,
	outPtr *C.double,
) C.int {
	tt := int(t)
	if tt == 0 {
		return 0
	}
	if int(window) < 1 {
		return -1
	}
	amps := unsafe.Slice((*float64)(unsafe.Pointer(ampsPtr)), tt)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), tt)
	extractEnvelope(amps, int(window), out)
	return 0
}

//export EnvelopeModulationDepth
func EnvelopeModulationDepth(
	envPtr *C.double,
	t C.int,
	outVal *C.double,
) C.int {
	tt := int(t)
	if tt == 0 {
		*outVal = 0.0
		return 0
	}
	env := unsafe.Slice((*float64)(unsafe.Pointer(envPtr)), tt)
	*outVal = C.double(envelopeModulationDepth(env))
	return 0
}

func main() {}
