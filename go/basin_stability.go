// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Steady-state R (Go port)

// Package main builds “libbasin_stability.so“ — one-trial
// Kuramoto post-step order parameter averaged over a finite window.
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
) bool {
	old := make([]float64, n)
	copy(old, phases)
	for i := 0; i < n; i++ {
		coupling := 0.0
		base := i * n
		thetaI := old[i]
		for j := 0; j < n; j++ {
			kIJ := knmFlat[base+j] * kScale
			if !finite(kIJ) {
				return false
			}
			if kIJ == 0.0 {
				continue
			}
			aIJ := alphaFlat[base+j]
			angle := old[j] - thetaI - aIJ
			if !finite(angle) {
				return false
			}
			coupling += kIJ * math.Sin(angle)
		}
		velocity := omegas[i] + coupling
		phases[i] = thetaI + dt*velocity
		if !finite(velocity) || !finite(phases[i]) {
			return false
		}
	}
	return true
}

func orderParameter(phases []float64) float64 {
	// The admitted exported trial always has a nonempty population.
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
	if !validDimensions(n, nTransient, nMeasure) || len(phasesInit) != n || len(omegas) != n || len(knmFlat) != n*n || len(alphaFlat) != n*n || !finite(kScale) || !finite(dt) || dt <= 0 {
		return math.NaN()
	}
	for _, values := range [][]float64{phasesInit, omegas, knmFlat, alphaFlat} {
		for _, value := range values {
			if !finite(value) {
				return math.NaN()
			}
		}
	}
	if nMeasure == 0 {
		return 0
	}
	phases := make([]float64, len(phasesInit))
	copy(phases, phasesInit)
	for s := 0; s < nTransient; s++ {
		if !kuramotoStep(phases, omegas, knmFlat, alphaFlat, n, kScale, dt) {
			return math.NaN()
		}
	}
	rSum := 0.0
	for s := 0; s < nMeasure; s++ {
		if !kuramotoStep(phases, omegas, knmFlat, alphaFlat, n, kScale, dt) {
			return math.NaN()
		}
		rSum += orderParameter(phases)
	}
	r := rSum / float64(nMeasure)
	if !finite(r) || r < 0 || r > 1+1e-12 {
		return math.NaN()
	}
	return math.Min(r, 1)
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func validDimensions(n, transient, measure int) bool {
	// unsafe.Slice requires both element and byte counts to fit int.
	maximum := int(^uint(0)>>1) / 8
	return n > 0 && n <= maximum/n && transient >= 0 && measure >= 0
}

// SteadyStateR retains the legacy ABI. Callers must supply n/n*n readable
// doubles; the ABI has no extent fields. Invalid scalar metadata returns NaN.
//
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
	if !validDimensions(nn, int(nTransient), int(nMeasure)) || phasesInitPtr == nil || omegasPtr == nil || knmPtr == nil || alphaPtr == nil {
		return C.double(math.NaN())
	}
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

// SteadyStateRV2 admits metadata before constructing slices. Extent fields
// are counts of readable doubles, whose storage remains the caller's obligation.
//
//export SteadyStateRV2
func SteadyStateRV2(
	phasesPtr, omegasPtr, knmPtr, alphaPtr *C.double,
	phasesLen, omegasLen, knmLen, alphaLen C.ulonglong,
	n C.longlong, scale, dt C.double, transient, measure C.longlong,
) C.double {
	maximum := int64(int(^uint(0) >> 1))
	if int64(n) <= 0 || int64(n) > maximum || int64(transient) < 0 || int64(transient) > maximum || int64(measure) < 0 || int64(measure) > maximum {
		return C.double(math.NaN())
	}
	nn := int(n)
	if !validDimensions(nn, int(transient), int(measure)) {
		return C.double(math.NaN())
	}
	if uint64(phasesLen) != uint64(nn) || uint64(omegasLen) != uint64(nn) || uint64(knmLen) != uint64(nn*nn) || uint64(alphaLen) != uint64(nn*nn) {
		return C.double(math.NaN())
	}
	if phasesPtr == nil || omegasPtr == nil || knmPtr == nil || alphaPtr == nil {
		return C.double(math.NaN())
	}
	return C.double(steadyStateR(
		unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn),
		unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn),
		unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn),
		unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn),
		nn, float64(scale), float64(dt), int(transient), int(measure),
	))
}

func main() {}
