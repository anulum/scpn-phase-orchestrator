// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Spectral graph eigendecomposition (Go port)

// Package main builds ``libspectral.so`` — symmetric eigendecomposition
// of the combinatorial graph Laplacian via ``gonum.org/v1/gonum/mat``.
//
// The exported C ABI returns both the eigenvalue vector and the
// Fiedler eigenvector (``λ₂``-eigenvector) through caller-preallocated
// output buffers.
//
// Build with::
//
//	go build -buildmode=c-shared -o libspectral.so spectral.go
package main

import "C"

import (
	"math"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

func spectralEig(
	knmFlat []float64,
	n int,
	outEigvals []float64,
	outFiedler []float64,
) int {
	// Build L = D - |W| with zeroed diagonal.
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			w := math.Abs(knmFlat[i*n+j])
			data[i*n+j] = -w
			sum += w
		}
		data[i*n+i] = sum
	}
	// Symmetrise explicitly — the (i, j) vs (j, i) paths may differ
	// by a tiny amount in the input knm, and gonum's EigenSym panics
	// on non-symmetric input.
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := 0.5 * (data[i*n+j] + data[j*n+i])
			sym.SetSym(i, j, v)
		}
	}

	var eig mat.EigenSym
	if !eig.Factorize(sym, true) {
		return 1
	}
	vals := eig.Values(nil)
	var vecs mat.Dense
	eig.VectorsTo(&vecs)

	for i := 0; i < n; i++ {
		outEigvals[i] = vals[i]
	}
	// Fiedler vector = column 1 (second smallest eigenvalue).
	for i := 0; i < n; i++ {
		outFiedler[i] = vecs.At(i, 1)
	}
	return 0
}

//export SpectralEig
func SpectralEig(
	knmPtr *C.double,
	n C.int,
	eigvalsPtr *C.double,
	fiedlerPtr *C.double,
) C.int {
	nn := int(n)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	eigvals := unsafe.Slice((*float64)(unsafe.Pointer(eigvalsPtr)), nn)
	fiedler := unsafe.Slice((*float64)(unsafe.Pointer(fiedlerPtr)), nn)
	return C.int(spectralEig(knm, nn, eigvals, fiedler))
}

func main() {}
