// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Fractal-dimension kernels (Go port)

// Package main builds “libdimension.so“ — Grassberger-Procaccia
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
			s = math.Hypot(s, delta)
		}
		dists[p] = s
	}
	for k := 0; k < nK; k++ {
		cnt := 0
		eps := epsilons[k]
		for p := 0; p < nP; p++ {
			if dists[p] < eps {
				cnt++
			}
		}
		out[k] = float64(cnt) / float64(nP)
	}
}

func kaplanYorkeDimension(le []float64) float64 {
	scale := 0.0
	for _, value := range le {
		if !finite(value) {
			return math.NaN()
		}
		scale = math.Max(scale, math.Abs(value))
	}
	if scale == 0.0 {
		return float64(len(le))
	}
	sorted := append([]float64(nil), le...)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	total, correction, accepted := 0.0, 0.0, 0.0
	for index, exponent := range sorted {
		value := exponent / scale
		updated := total + value
		if math.Abs(total) >= math.Abs(value) {
			correction += (total - updated) + value
		} else {
			correction += (value - updated) + total
		}
		total = updated
		prefix := total + correction
		if prefix < 0.0 {
			return float64(index) + accepted/math.Abs(value)
		}
		accepted = prefix
	}
	return float64(len(sorted))
}

// finite reports whether a numerical measurement belongs to the accepted domain.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// CorrelationIntegral writes nK fractions into outPtr.
// trajPtr: flat (T, d) row-major. idxI/idxJ: pair index arrays of
// length nP. epsPtr: nK thresholds.
//
//export CorrelationIntegral
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
	// Pointer capacity remains the C caller's obligation. Reject invalid metadata
	// and null pointers before creating any slices or writing output storage.
	if tt < 0 || dd <= 0 || np_ < 0 || nk < 0 || (tt > 0 && dd > int(^uint(0)>>1)/tt) {
		return -1
	}
	if (tt*dd > 0 && trajPtr == nil) || (np_ > 0 && (idxIPtr == nil || idxJPtr == nil)) ||
		(nk > 0 && (epsPtr == nil || outPtr == nil)) {
		return -1
	}
	traj := unsafe.Slice((*float64)(unsafe.Pointer(trajPtr)), tt*dd)
	idxI := unsafe.Slice((*int64)(unsafe.Pointer(idxIPtr)), np_)
	idxJ := unsafe.Slice((*int64)(unsafe.Pointer(idxJPtr)), np_)
	eps := unsafe.Slice((*float64)(unsafe.Pointer(epsPtr)), nk)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nk)
	for _, value := range traj {
		if !finite(value) {
			return -2
		}
	}
	for p, i := range idxI {
		j := idxJ[p]
		if i < 0 || j < 0 || i >= int64(tt) || j >= int64(tt) || i == j {
			return -3
		}
	}
	for k, value := range eps {
		if !finite(value) || value < 0.0 || (k > 0 && value < eps[k-1]) {
			return -4
		}
	}
	correlationIntegral(traj, tt, dd, idxI, idxJ, eps, out)
	return 0
}

// KaplanYorkeDimension writes a single f64 scalar into outVal.
//
//export KaplanYorkeDimension
func KaplanYorkeDimension(
	lePtr *C.double,
	n C.int,
	outVal *C.double,
) C.int {
	nn := int(n)
	if nn < 0 || outVal == nil || (nn > 0 && lePtr == nil) {
		return -1
	}
	le := unsafe.Slice((*float64)(unsafe.Pointer(lePtr)), nn)
	for _, value := range le {
		if !finite(value) {
			return -2
		}
	}
	*outVal = C.double(kaplanYorkeDimension(le))
	return 0
}

func main() {}
