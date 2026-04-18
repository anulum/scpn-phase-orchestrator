// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Poincaré section kernels (Go port)

// Package main builds ``libpoincare.so`` — a C-shared library
// exporting two Poincaré-section kernels with variable-length
// output: PoincareSection and PhasePoincare. Caller pre-allocates
// crossings + times buffers; the call returns the populated count.
//
// Build with::
//
//	go build -buildmode=c-shared -o libpoincare.so poincare.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiPoin = 2.0 * math.Pi

func unwrap(signal []float64) []float64 {
	out := make([]float64, len(signal))
	if len(signal) == 0 {
		return out
	}
	out[0] = signal[0]
	for i := 1; i < len(signal); i++ {
		diff := signal[i] - signal[i-1] + (out[i-1] - signal[i-1])
		if diff > math.Pi {
			delta := -twoPiPoin * math.Floor((diff+math.Pi)/twoPiPoin)
			out[i] = signal[i] + (out[i-1] - signal[i-1]) + delta
		} else if diff < -math.Pi {
			delta := twoPiPoin * math.Floor((-diff+math.Pi)/twoPiPoin)
			out[i] = signal[i] + (out[i-1] - signal[i-1]) + delta
		} else {
			out[i] = signal[i] + (out[i-1] - signal[i-1])
		}
	}
	return out
}

func poincareSection(
	traj []float64,
	t, d int,
	normal []float64,
	offset float64,
	directionID int,
	crossings []float64,
	times []float64,
) int {
	normSq := 0.0
	for _, v := range normal {
		normSq += v * v
	}
	normMag := math.Sqrt(normSq)
	if normMag <= 0.0 {
		return 0
	}
	nVec := make([]float64, d)
	for i := 0; i < d; i++ {
		nVec[i] = normal[i] / normMag
	}

	signed := make([]float64, t)
	for i := 0; i < t; i++ {
		s := 0.0
		base := i * d
		for k := 0; k < d; k++ {
			s += traj[base+k] * nVec[k]
		}
		signed[i] = s - offset
	}

	nCr := 0
	for i := 0; i < t-1; i++ {
		d0 := signed[i]
		d1 := signed[i+1]
		var isCross bool
		switch directionID {
		case 0:
			isCross = d0 < 0.0 && d1 >= 0.0
		case 1:
			isCross = d0 > 0.0 && d1 <= 0.0
		default:
			isCross = d0*d1 < 0.0
		}
		if !isCross {
			continue
		}
		alpha := 0.5
		if math.Abs(d1-d0) > 1e-15 {
			alpha = -d0 / (d1 - d0)
		}
		baseI := i * d
		baseNext := (i + 1) * d
		for k := 0; k < d; k++ {
			xi := traj[baseI+k]
			xj := traj[baseNext+k]
			crossings[nCr*d+k] = xi + alpha*(xj-xi)
		}
		times[nCr] = float64(i) + alpha
		nCr++
	}
	return nCr
}

func phasePoincare(
	phases []float64,
	t, n int,
	oscillatorIdx int,
	sectionPhase float64,
	crossings []float64,
	times []float64,
) int {
	target := make([]float64, t)
	for i := 0; i < t; i++ {
		target[i] = phases[i*n+oscillatorIdx]
	}
	unwrapped := unwrap(target)
	shifted := make([]float64, t)
	for i := 0; i < t; i++ {
		v := math.Mod(unwrapped[i]-sectionPhase, twoPiPoin)
		if v < 0 {
			v += twoPiPoin
		}
		shifted[i] = v
	}

	nCr := 0
	for i := 0; i < t-1; i++ {
		if shifted[i] > math.Pi && shifted[i+1] < math.Pi {
			denom := shifted[i] - shifted[i+1] + twoPiPoin
			alpha := 0.5
			if denom != 0.0 {
				alpha = shifted[i] / denom
			}
			if alpha < 0.0 {
				alpha = 0.0
			} else if alpha > 1.0 {
				alpha = 1.0
			}
			baseI := i * n
			baseNext := (i + 1) * n
			for k := 0; k < n; k++ {
				xi := phases[baseI+k]
				xj := phases[baseNext+k]
				crossings[nCr*n+k] = xi + alpha*(xj-xi)
			}
			times[nCr] = float64(i) + alpha
			nCr++
		}
	}
	return nCr
}

//export PoincareSection
func PoincareSection(
	trajPtr *C.double, t C.int, d C.int,
	normalPtr *C.double, offset C.double, directionID C.int,
	crossingsPtr *C.double, timesPtr *C.double,
) C.int {
	tt := int(t)
	dd := int(d)
	traj := unsafe.Slice((*float64)(unsafe.Pointer(trajPtr)), tt*dd)
	normal := unsafe.Slice((*float64)(unsafe.Pointer(normalPtr)), dd)
	crossings := unsafe.Slice((*float64)(unsafe.Pointer(crossingsPtr)), tt*dd)
	times := unsafe.Slice((*float64)(unsafe.Pointer(timesPtr)), tt)
	return C.int(poincareSection(
		traj, tt, dd, normal, float64(offset), int(directionID),
		crossings, times,
	))
}

//export PhasePoincare
func PhasePoincare(
	phasesPtr *C.double, t C.int, n C.int,
	oscillatorIdx C.int, sectionPhase C.double,
	crossingsPtr *C.double, timesPtr *C.double,
) C.int {
	tt := int(t)
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt*nn)
	crossings := unsafe.Slice((*float64)(unsafe.Pointer(crossingsPtr)), tt*nn)
	times := unsafe.Slice((*float64)(unsafe.Pointer(timesPtr)), tt)
	return C.int(phasePoincare(
		phases, tt, nn, int(oscillatorIdx), float64(sectionPhase),
		crossings, times,
	))
}

func main() {}
