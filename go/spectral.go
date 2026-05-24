// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Spectral graph eigendecomposition (Go port)

// Package main builds libspectral.so for symmetric eigendecomposition
// of the combinatorial graph Laplacian via gonum.org/v1/gonum/mat.
//
// The exported C ABI returns both the eigenvalue vector and the
// Fiedler eigenvector, the lambda-2 eigenvector, through caller-preallocated
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
	// Build L = D - A, where A_ij is the reciprocal undirected
	// magnitude weight (|W_ij| + |W_ji|) / 2 and A_ii = 0.
	data := make([]float64, n*n)
	degree := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			w := 0.5 * (math.Abs(knmFlat[i*n+j]) + math.Abs(knmFlat[j*n+i]))
			data[i*n+j] = -w
			data[j*n+i] = -w
			degree[i] += w
			degree[j] += w
		}
	}
	for i := 0; i < n; i++ {
		data[i*n+i] = degree[i]
	}
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, data[i*n+j])
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
