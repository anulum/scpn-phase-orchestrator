// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — spatial coupling modulation (Go port)

// Package main builds libspatial_modulator.so for distance-dependent
// coupling modulation.
//
// Build with:
//
//	go build -buildmode=c-shared -o libspatial_modulator.so spatial_modulator.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func spatialWeight(distance, kBase float64, form int, exponent, length, epsilon float64) float64 {
	switch form {
	case 0:
		return kBase / (1.0 + distance)
	case 1:
		return kBase * math.Exp(-distance/length)
	case 2:
		return kBase * math.Pow(1.0+distance/length, -exponent)
	case 3:
		return kBase / math.Sqrt(distance*distance+epsilon)
	default:
		return math.NaN()
	}
}

func spatialModulate(knm, positions []float64, n, dim int, kBase float64, form int, exponent, length, epsilon float64, out []float64) int {
	if n <= 0 || dim <= 0 || !isFinite(kBase) || !isFinite(exponent) || !isFinite(length) || !isFinite(epsilon) || exponent <= 0.0 || length <= 0.0 || epsilon <= 0.0 {
		return 1
	}
	if form < 0 || form > 3 {
		return 2
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			if i == j {
				out[idx] = 0.0
				continue
			}
			d2 := 0.0
			for d := 0; d < dim; d++ {
				delta := positions[i*dim+d] - positions[j*dim+d]
				d2 += delta * delta
			}
			distance := math.Sqrt(d2)
			weight := spatialWeight(distance, kBase, form, exponent, length, epsilon)
			if !isFinite(weight) {
				return 3
			}
			out[idx] = knm[idx] * weight
		}
	}
	return 0
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

//export SpatialModulate
func SpatialModulate(
	knmPtr *C.double,
	positionsPtr *C.double,
	n C.int,
	dim C.int,
	kBase C.double,
	form C.int,
	exponent C.double,
	length C.double,
	epsilon C.double,
	outPtr *C.double,
) C.int {
	nn := int(n)
	dd := int(dim)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	positions := unsafe.Slice((*float64)(unsafe.Pointer(positionsPtr)), nn*dd)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn)
	rc := spatialModulate(knm, positions, nn, dd, float64(kBase), int(form), float64(exponent), float64(length), float64(epsilon), out)
	return C.int(rc)
}

func main() {}
