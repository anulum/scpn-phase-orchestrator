// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ordinal-Pattern Transition Entropy (Go port)

// Package main builds ``libopt_entropy.so`` — a C-shared library exporting
// the ordinal-pattern transition entropy kernels. The algorithm matches the
// Rust, NumPy, Julia, and Mojo reference implementations bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libopt_entropy.so opt_entropy.go
package main

import "C"

import (
	"math"
	"sort"
	"unsafe"
)

func factorial(value int) int64 {
	var result int64 = 1
	for factor := 2; factor <= value; factor++ {
		result *= int64(factor)
	}
	return result
}

func windowCount(length, dimension, delay int) int {
	span := (dimension - 1) * delay
	if length > span {
		return length - span
	}
	return 0
}

func stableArgsort(window []float64) []int {
	dimension := len(window)
	used := make([]bool, dimension)
	perm := make([]int, dimension)
	for rank := 0; rank < dimension; rank++ {
		best := -1
		for idx := 0; idx < dimension; idx++ {
			if used[idx] {
				continue
			}
			if best == -1 || window[idx] < window[best] ||
				(window[idx] == window[best] && idx < best) {
				best = idx
			}
		}
		perm[rank] = best
		used[best] = true
	}
	return perm
}

func lehmerCode(perm []int, fact []int64) int64 {
	dimension := len(perm)
	var code int64
	for i := 0; i < dimension; i++ {
		var smaller int64
		for j := i + 1; j < dimension; j++ {
			if perm[j] < perm[i] {
				smaller++
			}
		}
		code += smaller * fact[dimension-1-i]
	}
	return code
}

func ordinalPatternSequence(series []float64, dimension, delay int) []int64 {
	count := windowCount(len(series), dimension, delay)
	fact := make([]int64, dimension)
	for k := 0; k < dimension; k++ {
		fact[k] = factorial(k)
	}
	codes := make([]int64, count)
	window := make([]float64, dimension)
	for m := 0; m < count; m++ {
		for k := 0; k < dimension; k++ {
			window[k] = series[m+k*delay]
		}
		codes[m] = lehmerCode(stableArgsort(window), fact)
	}
	return codes
}

func transitionEntropy(series []float64, dimension, delay int) float64 {
	codes := ordinalPatternSequence(series, dimension, delay)
	nCodes := len(codes)
	if nCodes < 2 {
		return 0.0
	}
	factD := factorial(dimension)
	total := nCodes - 1
	keys := make([]int64, total)
	for m := 0; m < total; m++ {
		keys[m] = codes[m]*factD + codes[m+1]
	}
	sort.Slice(keys, func(a, b int) bool { return keys[a] < keys[b] })

	var counts []int64
	run := int64(1)
	for idx := 1; idx < len(keys); idx++ {
		if keys[idx] == keys[idx-1] {
			run++
		} else {
			counts = append(counts, run)
			run = 1
		}
	}
	counts = append(counts, run)

	distinct := len(counts)
	if distinct < 2 {
		return 0.0
	}
	totalF := float64(total)
	entropy := 0.0
	for _, count := range counts {
		probability := float64(count) / totalF
		entropy -= probability * math.Log(probability)
	}
	maxEntropy := math.Log(float64(distinct))
	if maxEntropy < 1e-15 {
		return 0.0
	}
	value := entropy / maxEntropy
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

//export OrdinalPatternSequence
//
// OrdinalPatternSequence writes the Lehmer-encoded ordinal-pattern code
// sequence into outPtr (caller allocates windowCount entries). Returns 0.
func OrdinalPatternSequence(
	seriesPtr *C.double,
	n C.int,
	dimension C.int,
	delay C.int,
	outPtr *C.longlong,
) C.int {
	nn := int(n)
	series := unsafe.Slice((*float64)(unsafe.Pointer(seriesPtr)), nn)
	codes := ordinalPatternSequence(series, int(dimension), int(delay))
	if len(codes) == 0 {
		return 0
	}
	out := unsafe.Slice((*int64)(unsafe.Pointer(outPtr)), len(codes))
	copy(out, codes)
	return 0
}

//export TransitionEntropy
//
// TransitionEntropy writes the normalised transition entropy into outValue.
// Returns 0 on success.
func TransitionEntropy(
	seriesPtr *C.double,
	n C.int,
	dimension C.int,
	delay C.int,
	outValue *C.double,
) C.int {
	nn := int(n)
	series := unsafe.Slice((*float64)(unsafe.Pointer(seriesPtr)), nn)
	*outValue = C.double(transitionEntropy(series, int(dimension), int(delay)))
	return 0
}

func main() {}
