// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Financial market PLV / R(t) (Go port)

// Package main builds ``libmarket.so`` exporting the per-row
// order parameter and the windowed phase-locking-value matrix
// computed with the Rust kernel's sincos expansion.
//
// Build with::
//
//	go build -buildmode=c-shared -o libmarket.so market.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func orderParameter(
	phasesFlat []float64,
	t, n int,
	out []float64,
) {
	invN := 1.0 / float64(n)
	for row := 0; row < t; row++ {
		sumCos := 0.0
		sumSin := 0.0
		base := row * n
		for i := 0; i < n; i++ {
			theta := phasesFlat[base+i]
			sumCos += math.Cos(theta)
			sumSin += math.Sin(theta)
		}
		mc := sumCos * invN
		ms := sumSin * invN
		out[row] = math.Sqrt(mc*mc + ms*ms)
	}
}

func plvMatrix(
	phasesFlat []float64,
	t, n, window int,
	out []float64,
) {
	nWindows := t - window + 1
	invW := 1.0 / float64(window)
	windowS := make([]float64, window*n)
	windowC := make([]float64, window*n)
	for w := 0; w < nWindows; w++ {
		for k := 0; k < window; k++ {
			baseStep := (w + k) * n
			for i := 0; i < n; i++ {
				theta := phasesFlat[baseStep+i]
				windowS[k*n+i] = math.Sin(theta)
				windowC[k*n+i] = math.Cos(theta)
			}
		}
		matOffset := w * n * n
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				sumCos := 0.0
				sumSin := 0.0
				for k := 0; k < window; k++ {
					si := windowS[k*n+i]
					ci := windowC[k*n+i]
					sj := windowS[k*n+j]
					cj := windowC[k*n+j]
					sumCos += cj*ci + sj*si
					sumSin += sj*ci - cj*si
				}
				mc := sumCos * invW
				ms := sumSin * invW
				out[matOffset+i*n+j] = math.Sqrt(mc*mc + ms*ms)
			}
		}
	}
}

//export MarketOrderParameter
func MarketOrderParameter(
	phasesPtr *C.double,
	t, n C.int,
	outPtr *C.double,
) C.int {
	tt := int(t)
	nn := int(n)
	if tt == 0 || nn == 0 {
		return 0
	}
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), tt)
	orderParameter(phases, tt, nn, out)
	return 0
}

//export MarketPLV
func MarketPLV(
	phasesPtr *C.double,
	t, n, window C.int,
	outPtr *C.double,
) C.int {
	tt := int(t)
	nn := int(n)
	ww := int(window)
	if tt < ww || nn == 0 || ww == 0 {
		return 0
	}
	nWindows := tt - ww + 1
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt*nn)
	out := unsafe.Slice(
		(*float64)(unsafe.Pointer(outPtr)), nWindows*nn*nn,
	)
	plvMatrix(phases, tt, nn, ww, out)
	return 0
}

func main() {}
