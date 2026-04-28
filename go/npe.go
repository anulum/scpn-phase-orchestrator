// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Normalised Persistent Entropy (Go port)

// Package main builds ``libnpe.so`` — a C-shared library exporting
// the NPE + phase distance matrix kernels. Algorithm matches the Rust
// and NumPy reference implementations bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libnpe.so npe.go
package main

import "C"

import (
	"math"
	"sort"
	"unsafe"
)

type edge struct {
	d float64
	i int
	j int
}

func phaseDistanceMatrix(phases []float64) []float64 {
	n := len(phases)
	out := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d := phases[i] - phases[j]
			out[i*n+j] = math.Abs(math.Atan2(math.Sin(d), math.Cos(d)))
		}
	}
	return out
}

func findRoot(parent []int, x int) int {
	for parent[x] != x {
		parent[x] = parent[parent[x]]
		x = parent[x]
	}
	return x
}

func computeNPE(phases []float64, maxRadius float64) float64 {
	n := len(phases)
	if n < 2 {
		return 0.0
	}
	radius := maxRadius
	if radius < 0.0 {
		radius = math.Pi
	}
	dist := phaseDistanceMatrix(phases)

	edges := make([]edge, 0, n*(n-1)/2)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			edges = append(edges, edge{dist[i*n+j], i, j})
		}
	}
	sort.Slice(edges, func(a, b int) bool { return edges[a].d < edges[b].d })

	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	rankArr := make([]int, n)
	var lifetimes []float64
	for _, e := range edges {
		if e.d > radius {
			break
		}
		ri := findRoot(parent, e.i)
		rj := findRoot(parent, e.j)
		if ri != rj {
			lifetimes = append(lifetimes, e.d)
			if rankArr[ri] < rankArr[rj] {
				parent[ri] = rj
			} else if rankArr[ri] > rankArr[rj] {
				parent[rj] = ri
			} else {
				parent[rj] = ri
				rankArr[ri]++
			}
		}
	}
	if len(lifetimes) == 0 {
		return 0.0
	}
	total := 0.0
	for _, lt := range lifetimes {
		total += lt
	}
	if total < 1e-15 {
		return 0.0
	}
	entropy := 0.0
	nProbs := 0
	for _, lt := range lifetimes {
		p := lt / total
		if p > 0.0 {
			entropy -= p * math.Log(p)
			nProbs++
		}
	}
	var maxEntropy float64
	if nProbs > 1 {
		maxEntropy = math.Log(float64(nProbs))
	} else {
		maxEntropy = 1.0
	}
	if maxEntropy < 1e-15 {
		return 0.0
	}
	return entropy / maxEntropy
}

//export PhaseDistanceMatrix
//
// PhaseDistanceMatrix writes a flat row-major (N, N) distance matrix
// into outPtr (caller allocates n*n capacity).
func PhaseDistanceMatrix(
	phasesPtr *C.double,
	n C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn)
	result := phaseDistanceMatrix(phases)
	copy(out, result)
	return 0
}

//export ComputeNPE
//
// ComputeNPE writes the NPE into outNPE. Returns 0 on success.
func ComputeNPE(
	phasesPtr *C.double,
	n C.int,
	maxRadius C.double,
	outNPE *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	*outNPE = C.double(computeNPE(phases, float64(maxRadius)))
	return 0
}

func main() {}
