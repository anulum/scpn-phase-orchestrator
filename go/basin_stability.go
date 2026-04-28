// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Steady-state R (Go port)

// Package main builds ``libbasin_stability.so`` — one-trial
// Kuramoto steady-state order parameter averaged over a window.
//
// Build with::
//
//	go build -buildmode=c-shared -o libbasin_stability.so basin_stability.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func kuramotoStep(
	phases, omegas, knmFlat, alphaFlat []float64,
	n int,
	kScale, dt float64,
) {
	old := make([]float64, n)
	copy(old, phases)
	for i := 0; i < n; i++ {
		coupling := 0.0
		base := i * n
		thetaI := old[i]
		for j := 0; j < n; j++ {
			kIJ := knmFlat[base+j] * kScale
			if math.Abs(kIJ) < 1e-30 {
				continue
			}
			aIJ := alphaFlat[base+j]
			coupling += kIJ * math.Sin(old[j]-thetaI-aIJ)
		}
		phases[i] = thetaI + dt*(omegas[i]+coupling)
	}
}

func orderParameter(phases []float64) float64 {
	if len(phases) == 0 {
		return 0.0
	}
	n := float64(len(phases))
	sumCos := 0.0
	sumSin := 0.0
	for _, t := range phases {
		sumCos += math.Cos(t)
		sumSin += math.Sin(t)
	}
	return math.Sqrt(math.Pow(sumCos/n, 2) + math.Pow(sumSin/n, 2))
}

func steadyStateR(
	phasesInit, omegas, knmFlat, alphaFlat []float64,
	n int,
	kScale, dt float64,
	nTransient, nMeasure int,
) float64 {
	phases := make([]float64, len(phasesInit))
	copy(phases, phasesInit)
	for s := 0; s < nTransient; s++ {
		kuramotoStep(phases, omegas, knmFlat, alphaFlat, n, kScale, dt)
	}
	rSum := 0.0
	for s := 0; s < nMeasure; s++ {
		kuramotoStep(phases, omegas, knmFlat, alphaFlat, n, kScale, dt)
		rSum += orderParameter(phases)
	}
	if nMeasure == 0 {
		return 0.0
	}
	return rSum / float64(nMeasure)
}

//export SteadyStateR
func SteadyStateR(
	phasesInitPtr *C.double,
	omegasPtr *C.double,
	knmPtr *C.double,
	alphaPtr *C.double,
	n C.int,
	kScale C.double,
	dt C.double,
	nTransient C.int,
	nMeasure C.int,
) C.double {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesInitPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	r := steadyStateR(
		phases, omegas, knm, alpha,
		nn, float64(kScale), float64(dt),
		int(nTransient), int(nMeasure),
	)
	return C.double(r)
}

func main() {}
