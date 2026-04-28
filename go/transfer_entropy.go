// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Transfer entropy (Go port)

// Package main builds libtransfer_entropy.so.
//
// Build with::
//
//	go build -buildmode=c-shared -o libtransfer_entropy.so transfer_entropy.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPi = 2.0 * math.Pi

func conditionalEntropy(target []int, condition []int, nCondBins int) float64 {
	n := len(target)
	if n == 0 {
		return 0.0
	}
	// Group target values by condition bin.
	groups := make(map[int][]int, nCondBins)
	for i := 0; i < n; i++ {
		c := condition[i]
		groups[c] = append(groups[c], target[i])
	}
	h := 0.0
	for c := 0; c < nCondBins; c++ {
		vals, ok := groups[c]
		if !ok {
			continue
		}
		count := len(vals)
		if count < 2 {
			continue
		}
		counts := make(map[int]int, 8)
		for _, v := range vals {
			counts[v]++
		}
		sub := 0.0
		for _, cnt := range counts {
			p := float64(cnt) / float64(count)
			sub += p * math.Log(p+1e-30)
		}
		h -= float64(count) / float64(n) * sub
	}
	return h
}

func phaseTE(source, target []float64, nBins int) float64 {
	if len(source) < 3 || len(target) < 3 {
		return 0.0
	}
	n := len(source)
	if len(target) < n {
		n = len(target)
	}
	n -= 1
	binWidth := twoPi / float64(nBins)
	srcBinned := make([]int, n)
	tgtBinned := make([]int, n)
	tgtNext := make([]int, n)
	for i := 0; i < n; i++ {
		s := math.Mod(source[i], twoPi)
		if s < 0 {
			s += twoPi
		}
		t := math.Mod(target[i], twoPi)
		if t < 0 {
			t += twoPi
		}
		tn := math.Mod(target[i+1], twoPi)
		if tn < 0 {
			tn += twoPi
		}
		srcBinned[i] = int(s / binWidth)
		if srcBinned[i] >= nBins {
			srcBinned[i] = nBins - 1
		}
		tgtBinned[i] = int(t / binWidth)
		if tgtBinned[i] >= nBins {
			tgtBinned[i] = nBins - 1
		}
		tgtNext[i] = int(tn / binWidth)
		if tgtNext[i] >= nBins {
			tgtNext[i] = nBins - 1
		}
	}
	hYtYt := conditionalEntropy(tgtNext, tgtBinned, nBins)
	jointCond := make([]int, n)
	for i := 0; i < n; i++ {
		jointCond[i] = tgtBinned[i]*nBins + srcBinned[i]
	}
	hYtYtXt := conditionalEntropy(tgtNext, jointCond, nBins*nBins)
	te := hYtYt - hYtYtXt
	if te < 0.0 {
		te = 0.0
	}
	return te
}

//export PhaseTransferEntropy
func PhaseTransferEntropy(
	srcPtr *C.double,
	tgtPtr *C.double,
	n C.int,
	nBins C.int,
	outTE *C.double,
) C.int {
	nn := int(n)
	src := unsafe.Slice((*float64)(unsafe.Pointer(srcPtr)), nn)
	tgt := unsafe.Slice((*float64)(unsafe.Pointer(tgtPtr)), nn)
	*outTE = C.double(phaseTE(src, tgt, int(nBins)))
	return 0
}

//export TransferEntropyMatrix
//
// phaseSeries is flat (nOsc * nTime) row-major in (oscillator, time);
// outPtr expects nOsc * nOsc cells.
func TransferEntropyMatrix(
	phaseSeriesPtr *C.double,
	nOsc C.int,
	nTime C.int,
	nBins C.int,
	outPtr *C.double,
) C.int {
	nO := int(nOsc)
	nT := int(nTime)
	series := unsafe.Slice((*float64)(unsafe.Pointer(phaseSeriesPtr)), nO*nT)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nO*nO)

	src := make([]float64, nT)
	tgt := make([]float64, nT)
	for i := 0; i < nO; i++ {
		for s := 0; s < nT; s++ {
			src[s] = series[i*nT+s]
		}
		for j := 0; j < nO; j++ {
			if i == j {
				continue
			}
			for s := 0; s < nT; s++ {
				tgt[s] = series[j*nT+s]
			}
			out[i*nO+j] = phaseTE(src, tgt, int(nBins))
		}
	}
	return 0
}

func main() {}
