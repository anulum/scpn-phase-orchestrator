// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Digital-twin divergence (Go port)

// Package main builds ``libtwin_confidence.so`` — a C-shared library
// exporting the digital-twin divergence kernel (phase Jensen–Shannon
// divergence plus order-parameter Wasserstein-1 distance). Matches the
// NumPy / Rust / Julia / Mojo references to 1e-9.
//
// Build with::
//
//	go build -buildmode=c-shared -o libtwin_confidence.so twin_confidence.go
package main

import "C"

import (
	"math"
	"sort"
	"unsafe"
)

const twoPi = 2.0 * math.Pi

func phaseHistogram(phases []float64, nBins int) []float64 {
	width := twoPi / float64(nBins)
	counts := make([]float64, nBins)
	for _, phase := range phases {
		wrapped := phase - math.Floor(phase/twoPi)*twoPi
		idx := int(math.Floor(wrapped / width))
		if idx < 0 {
			idx = 0
		}
		if idx > nBins-1 {
			idx = nBins - 1
		}
		counts[idx] += 1.0
	}
	total := 0.0
	for _, c := range counts {
		total += c
	}
	out := make([]float64, nBins)
	if total <= 0.0 {
		for i := range out {
			out[i] = 1.0 / float64(nBins)
		}
		return out
	}
	for i, c := range counts {
		out[i] = c / total
	}
	return out
}

func kl(p, m []float64) float64 {
	sum := 0.0
	for i, pi := range p {
		if pi > 0.0 {
			sum += pi * math.Log(pi/m[i])
		}
	}
	return sum
}

func jensenShannon(p, q []float64) float64 {
	m := make([]float64, len(p))
	for i := range p {
		m[i] = 0.5 * (p[i] + q[i])
	}
	return 0.5*kl(p, m) + 0.5*kl(q, m)
}

func wasserstein1(modelOrder, observedOrder []float64) float64 {
	sortedModel := make([]float64, len(modelOrder))
	sortedObs := make([]float64, len(observedOrder))
	copy(sortedModel, modelOrder)
	copy(sortedObs, observedOrder)
	sort.Float64s(sortedModel)
	sort.Float64s(sortedObs)
	sum := 0.0
	for i := range sortedModel {
		sum += math.Abs(sortedModel[i] - sortedObs[i])
	}
	return sum / float64(len(sortedModel))
}

//export TwinDivergence
//
// TwinDivergence writes [js, w1] into outPtr (two f64 entries).
// modelPhasesPtr / observedPhasesPtr are length-n vectors; modelOrderPtr /
// observedOrderPtr are length-w vectors. Returns 0 on success, non-zero on a
// contract violation.
func TwinDivergence(
	modelPhasesPtr *C.double,
	observedPhasesPtr *C.double,
	modelOrderPtr *C.double,
	observedOrderPtr *C.double,
	n C.int,
	w C.int,
	nBins C.int,
	outPtr *C.double,
) C.int {
	nInt := int(n)
	wInt := int(w)
	nBinsInt := int(nBins)
	if nInt < 1 || wInt < 1 || nBinsInt < 1 {
		return 1
	}
	modelPhases := unsafe.Slice((*float64)(unsafe.Pointer(modelPhasesPtr)), nInt)
	observedPhases := unsafe.Slice((*float64)(unsafe.Pointer(observedPhasesPtr)), nInt)
	modelOrder := unsafe.Slice((*float64)(unsafe.Pointer(modelOrderPtr)), wInt)
	observedOrder := unsafe.Slice((*float64)(unsafe.Pointer(observedOrderPtr)), wInt)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), 2)

	p := phaseHistogram(modelPhases, nBinsInt)
	q := phaseHistogram(observedPhases, nBinsInt)
	out[0] = jensenShannon(p, q)
	out[1] = wasserstein1(modelOrder, observedOrder)
	return 0
}

func main() {}
