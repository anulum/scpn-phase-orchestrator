// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Embedding primitives (Go port)

// Package main builds ``libembedding.so`` — a C-shared library
// exporting three delay-embedding primitives:
//
//   - DelayEmbed
//   - MutualInformation
//   - NearestNeighborDistances
//
// Build with::
//
//	go build -buildmode=c-shared -o libembedding.so embedding.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func delayEmbed(signal []float64, delay, dimension int, out []float64) {
	tEff := len(signal) - (dimension-1)*delay
	for i := 0; i < tEff; i++ {
		for d := 0; d < dimension; d++ {
			out[i*dimension+d] = signal[i+d*delay]
		}
	}
}

//export DelayEmbed
func DelayEmbed(
	signalPtr *C.double,
	tTotal C.int,
	delay C.int,
	dimension C.int,
	outPtr *C.double,
) C.int {
	n := int(tTotal)
	d := int(dimension)
	de := int(delay)
	tEff := n - (d-1)*de
	if tEff <= 0 {
		return -1
	}
	signal := unsafe.Slice((*float64)(unsafe.Pointer(signalPtr)), n)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), tEff*d)
	delayEmbed(signal, de, d, out)
	return 0
}

func mutualInformation(signal []float64, lag, nBins int) float64 {
	tTotal := len(signal) - lag
	if tTotal <= 0 {
		return 0.0
	}
	xMin, xMax := signal[0], signal[0]
	for i := 1; i < tTotal; i++ {
		v := signal[i]
		if v < xMin {
			xMin = v
		}
		if v > xMax {
			xMax = v
		}
	}
	yMin, yMax := signal[lag], signal[lag]
	for i := lag + 1; i < lag+tTotal; i++ {
		v := signal[i]
		if v < yMin {
			yMin = v
		}
		if v > yMax {
			yMax = v
		}
	}
	if xMax <= xMin || yMax <= yMin {
		return 0.0
	}
	dx := (xMax - xMin) / float64(nBins)
	dy := (yMax - yMin) / float64(nBins)
	hist := make([]float64, nBins*nBins)
	for i := 0; i < tTotal; i++ {
		x := signal[i]
		y := signal[i+lag]
		bx := int(math.Floor((x - xMin) / dx))
		by := int(math.Floor((y - yMin) / dy))
		if bx >= nBins {
			bx = nBins - 1
		}
		if by >= nBins {
			by = nBins - 1
		}
		hist[bx*nBins+by] += 1.0
	}
	total := float64(tTotal)
	hx := make([]float64, nBins)
	hy := make([]float64, nBins)
	for i := 0; i < nBins; i++ {
		for j := 0; j < nBins; j++ {
			h := hist[i*nBins+j]
			hx[i] += h
			hy[j] += h
		}
	}
	mi := 0.0
	for i := 0; i < nBins; i++ {
		for j := 0; j < nBins; j++ {
			h := hist[i*nBins+j]
			if h > 0 && hx[i] > 0 && hy[j] > 0 {
				pXY := h / total
				pX := hx[i] / total
				pY := hy[j] / total
				mi += pXY * math.Log(pXY/(pX*pY))
			}
		}
	}
	return mi
}

//export MutualInformation
func MutualInformation(
	signalPtr *C.double,
	tTotal C.int,
	lag C.int,
	nBins C.int,
	outVal *C.double,
) C.int {
	n := int(tTotal)
	signal := unsafe.Slice((*float64)(unsafe.Pointer(signalPtr)), n)
	*outVal = C.double(mutualInformation(signal, int(lag), int(nBins)))
	return 0
}

func nearestNeighborDistances(
	embedded []float64, t, m int, nnDist []float64, nnIdx []int64,
) {
	for i := 0; i < t; i++ {
		best := math.Inf(1)
		bestJ := int64(0)
		baseI := i * m
		for j := 0; j < t; j++ {
			if j == i {
				continue
			}
			baseJ := j * m
			d := 0.0
			for k := 0; k < m; k++ {
				delta := embedded[baseI+k] - embedded[baseJ+k]
				d += delta * delta
			}
			if d < best {
				best = d
				bestJ = int64(j)
			}
		}
		nnDist[i] = math.Sqrt(best)
		nnIdx[i] = bestJ
	}
}

//export NearestNeighborDistances
func NearestNeighborDistances(
	embeddedPtr *C.double,
	t C.int,
	m C.int,
	distPtr *C.double,
	idxPtr *C.longlong,
) C.int {
	tt := int(t)
	mm := int(m)
	embedded := unsafe.Slice((*float64)(unsafe.Pointer(embeddedPtr)), tt*mm)
	dist := unsafe.Slice((*float64)(unsafe.Pointer(distPtr)), tt)
	idx := unsafe.Slice((*int64)(unsafe.Pointer(idxPtr)), tt)
	nearestNeighborDistances(embedded, tt, mm, dist, idx)
	return 0
}

func main() {}
