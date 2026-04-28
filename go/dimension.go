// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Fractal-dimension kernels (Go port)

// Package main builds ``libdimension.so`` — Grassberger-Procaccia
// correlation integral + Kaplan-Yorke dimension. Python owns the
// RNG / pair subsampling for cross-backend parity.
//
// Build with::
//
//	go build -buildmode=c-shared -o libdimension.so dimension.go
package main

import "C"

import (
	"math"
	"sort"
	"unsafe"
)

func correlationIntegral(
	traj []float64,
	t, d int,
	idxI, idxJ []int64,
	epsilons []float64,
	out []float64,
) {
	nP := len(idxI)
	nK := len(epsilons)
	if nP == 0 {
		for k := 0; k < nK; k++ {
			out[k] = 0.0
		}
		return
	}
	dists := make([]float64, nP)
	for p := 0; p < nP; p++ {
		i := int(idxI[p])
		j := int(idxJ[p])
		s := 0.0
		baseI := i * d
		baseJ := j * d
		for k := 0; k < d; k++ {
			delta := traj[baseI+k] - traj[baseJ+k]
			s += delta * delta
		}
		dists[p] = math.Sqrt(s)
	}
	invP := 1.0 / float64(nP)
	for k := 0; k < nK; k++ {
		cnt := 0
		eps := epsilons[k]
		for p := 0; p < nP; p++ {
			if dists[p] < eps {
				cnt++
			}
		}
		out[k] = float64(cnt) * invP
	}
}

func kaplanYorkeDimension(le []float64) float64 {
	n := len(le)
	if n == 0 {
		return 0.0
	}
	sorted := make([]float64, n)
	copy(sorted, le)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	cumsum := 0.0
	j := -1
	for i := 0; i < n; i++ {
		cumsum += sorted[i]
		if cumsum >= 0.0 {
			j = i
		} else {
			break
		}
	}
	if j == -1 {
		return 0.0
	}
	if j >= n-1 {
		return float64(n)
	}
	denom := math.Abs(sorted[j+1])
	if denom == 0.0 {
		return float64(j + 1)
	}
	sJ := 0.0
	for i := 0; i <= j; i++ {
		sJ += sorted[i]
	}
	return float64(j+1) + sJ/denom
}

//export CorrelationIntegral
//
// CorrelationIntegral writes nK fractions into outPtr.
// trajPtr: flat (T, d) row-major. idxI/idxJ: pair index arrays of
// length nP. epsPtr: nK thresholds.
func CorrelationIntegral(
	trajPtr *C.double,
	t C.int,
	d C.int,
	idxIPtr *C.longlong,
	idxJPtr *C.longlong,
	nP C.int,
	epsPtr *C.double,
	nK C.int,
	outPtr *C.double,
) C.int {
	tt := int(t)
	dd := int(d)
	np_ := int(nP)
	nk := int(nK)
	traj := unsafe.Slice((*float64)(unsafe.Pointer(trajPtr)), tt*dd)
	idxI := unsafe.Slice((*int64)(unsafe.Pointer(idxIPtr)), np_)
	idxJ := unsafe.Slice((*int64)(unsafe.Pointer(idxJPtr)), np_)
	eps := unsafe.Slice((*float64)(unsafe.Pointer(epsPtr)), nk)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nk)
	correlationIntegral(traj, tt, dd, idxI, idxJ, eps, out)
	return 0
}

//export KaplanYorkeDimension
//
// KaplanYorkeDimension writes a single f64 scalar into outVal.
func KaplanYorkeDimension(
	lePtr *C.double,
	n C.int,
	outVal *C.double,
) C.int {
	nn := int(n)
	le := unsafe.Slice((*float64)(unsafe.Pointer(lePtr)), nn)
	*outVal = C.double(kaplanYorkeDimension(le))
	return 0
}

func main() {}
