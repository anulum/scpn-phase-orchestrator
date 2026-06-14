// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Combinatorial Hodge decomposition (Go port)

// Package main builds “libhodge.so“ — the combinatorial
// (Helmholtz–Hodge) decomposition of the Kuramoto coupling current
// into gradient, curl, and harmonic edge flows.
//
// The alternating edge flow is f_ij = 0.5*(K_ij + K_ji)*sin(theta_j -
// theta_i). With node-edge incidence B1 and edge-triangle incidence
// B2: f_grad = B1^T L0^+ (B1 f), f_curl = B2 L2^+ (B2^T f),
// f_harm = f - f_grad - f_curl. L0 = B1 B1^T and L2 = B2^T B2 are
// symmetric PSD; the pseudoinverse uses a symmetric eigendecomposition
// with a shared relative cutoff. Outputs are three flattened row-major
// N*N antisymmetric matrices.
//
// Build with::
//
//	go build -buildmode=c-shared -o libhodge.so hodge.go
package main

import "C"

import (
	"math"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

const pinvRcond = 1e-9

// psdPinvApply returns pinv(data) @ vec for a symmetric PSD row-major
// dim*dim matrix via eigendecomposition. Eigenvalues at or below
// pinvRcond*lambdaMax are treated as the null space. On factorisation
// failure the result is filled with NaN so the caller fails closed.
func psdPinvApply(data []float64, dim int, vec []float64) []float64 {
	out := make([]float64, dim)
	if dim == 0 {
		return out
	}
	sym := mat.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		for j := i; j < dim; j++ {
			sym.SetSym(i, j, data[i*dim+j])
		}
	}
	var eig mat.EigenSym
	if !eig.Factorize(sym, true) {
		for i := range out {
			out[i] = math.NaN()
		}
		return out
	}
	vals := eig.Values(nil)
	var vecs mat.Dense
	eig.VectorsTo(&vecs)
	lambdaMax := vals[dim-1]
	cutoff := 0.0
	if lambdaMax > 0.0 {
		cutoff = pinvRcond * lambdaMax
	}
	projected := make([]float64, dim)
	for k := 0; k < dim; k++ {
		if vals[k] <= cutoff {
			continue
		}
		acc := 0.0
		for r := 0; r < dim; r++ {
			acc += vecs.At(r, k) * vec[r]
		}
		projected[k] = acc / vals[k]
	}
	for r := 0; r < dim; r++ {
		acc := 0.0
		for k := 0; k < dim; k++ {
			acc += vecs.At(r, k) * projected[k]
		}
		out[r] = acc
	}
	return out
}

type signedEdge struct {
	edge int
	sign float64
}

func hodgeDecomposition(
	knm, phases []float64, n int,
	edges, tris []int64, nEdges, nTris int,
	gradOut, curlOut, harmOut []float64,
) {
	edgeI := make([]int, nEdges)
	edgeJ := make([]int, nEdges)
	flow := make([]float64, nEdges)
	for e := 0; e < nEdges; e++ {
		i := int(edges[2*e])
		j := int(edges[2*e+1])
		edgeI[e] = i
		edgeJ[e] = j
		kSym := 0.5 * (knm[i*n+j] + knm[j*n+i])
		flow[e] = kSym * math.Sin(phases[j]-phases[i])
	}

	l0 := make([]float64, n*n)
	div := make([]float64, n)
	for e := 0; e < nEdges; e++ {
		i := edgeI[e]
		j := edgeJ[e]
		l0[i*n+i] += 1.0
		l0[j*n+j] += 1.0
		l0[i*n+j] -= 1.0
		l0[j*n+i] -= 1.0
		div[i] -= flow[e]
		div[j] += flow[e]
	}
	potential := psdPinvApply(l0, n, div)
	fGrad := make([]float64, nEdges)
	for e := 0; e < nEdges; e++ {
		fGrad[e] = potential[edgeJ[e]] - potential[edgeI[e]]
	}

	fCurl := make([]float64, nEdges)
	if nTris > 0 {
		emap := make(map[[2]int]int, nEdges)
		for e := 0; e < nEdges; e++ {
			emap[[2]int{edgeI[e], edgeJ[e]}] = e
		}
		triEdges := make([][3]signedEdge, nTris)
		for t := 0; t < nTris; t++ {
			i := int(tris[3*t])
			j := int(tris[3*t+1])
			k := int(tris[3*t+2])
			triEdges[t] = [3]signedEdge{
				{emap[[2]int{i, j}], 1.0},
				{emap[[2]int{j, k}], 1.0},
				{emap[[2]int{i, k}], -1.0},
			}
		}
		l2 := make([]float64, nTris*nTris)
		c2 := make([]float64, nTris)
		for t := 0; t < nTris; t++ {
			for _, a := range triEdges[t] {
				c2[t] += a.sign * flow[a.edge]
			}
			for u := 0; u < nTris; u++ {
				acc := 0.0
				for _, a := range triEdges[t] {
					for _, b := range triEdges[u] {
						if a.edge == b.edge {
							acc += a.sign * b.sign
						}
					}
				}
				l2[t*nTris+u] = acc
			}
		}
		triPot := psdPinvApply(l2, nTris, c2)
		for t := 0; t < nTris; t++ {
			for _, a := range triEdges[t] {
				fCurl[a.edge] += a.sign * triPot[t]
			}
		}
	}

	for e := 0; e < nEdges; e++ {
		i := edgeI[e]
		j := edgeJ[e]
		g := fGrad[e]
		c := fCurl[e]
		hh := flow[e] - g - c
		gradOut[i*n+j] = g
		gradOut[j*n+i] = -g
		curlOut[i*n+j] = c
		curlOut[j*n+i] = -c
		harmOut[i*n+j] = hh
		harmOut[j*n+i] = -hh
	}
}

//export HodgeDecomposition
func HodgeDecomposition(
	knmPtr, phasesPtr *C.double, n C.int,
	edgesPtr *C.longlong, nEdges C.int,
	trisPtr *C.longlong, nTris C.int,
	gPtr, cPtr, hPtr *C.double,
) C.int {
	nn := int(n)
	ne := int(nEdges)
	nt := int(nTris)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	var edges []int64
	if ne > 0 {
		edges = unsafe.Slice((*int64)(unsafe.Pointer(edgesPtr)), 2*ne)
	}
	var tris []int64
	if nt > 0 {
		tris = unsafe.Slice((*int64)(unsafe.Pointer(trisPtr)), 3*nt)
	}
	grad := unsafe.Slice((*float64)(unsafe.Pointer(gPtr)), nn*nn)
	curl := unsafe.Slice((*float64)(unsafe.Pointer(cPtr)), nn*nn)
	harm := unsafe.Slice((*float64)(unsafe.Pointer(hPtr)), nn*nn)
	hodgeDecomposition(knm, phases, nn, edges, tris, ne, nt, grad, curl, harm)
	return 0
}

func main() {}
