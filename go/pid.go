// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Partial Information Decomposition (Go port)

// Package main builds libpid.so — the time-series Williams & Beer partial
// information decomposition of two oscillator groups about the global
// synchronisation state. The caller passes a row-major (t, n) phase history and
// 0-based group index arrays; the call writes (redundancy, synergy).
//
// Build with::
//
//	go build -buildmode=c-shared -o libpid.so pid.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const tauPid = 2.0 * math.Pi

func binAngle(angle float64, nBins int) int {
	w := math.Mod(angle, tauPid)
	if w < 0 {
		w += tauPid
	}
	b := int(math.Floor(w / (tauPid / float64(nBins))))
	if b > nBins-1 {
		b = nBins - 1
	}
	return b
}

func groupPhase(history []float64, n, row int, members []int) float64 {
	sinSum := 0.0
	cosSum := 0.0
	for _, j := range members {
		theta := history[row*n+j]
		sinSum += math.Sin(theta)
		cosSum += math.Cos(theta)
	}
	count := float64(len(members))
	return math.Atan2(sinSum/count, cosSum/count)
}

func globalPhase(history []float64, n, row int) float64 {
	sinSum := 0.0
	cosSum := 0.0
	for j := 0; j < n; j++ {
		theta := history[row*n+j]
		sinSum += math.Sin(theta)
		cosSum += math.Cos(theta)
	}
	count := float64(n)
	return math.Atan2(sinSum/count, cosSum/count)
}

func mutualInformation(joint, margX, margY []float64, nX, nBins int, total float64) float64 {
	if total <= 0.0 {
		return 0.0
	}
	mi := 0.0
	for x := 0; x < nX; x++ {
		if margX[x] <= 0.0 {
			continue
		}
		for y := 0; y < nBins; y++ {
			cxy := joint[x*nBins+y]
			if cxy <= 0.0 || margY[y] <= 0.0 {
				continue
			}
			pXY := cxy / total
			mi += pXY * math.Log(pXY/((margX[x]/total)*(margY[y]/total)))
		}
	}
	if mi < 0.0 {
		return 0.0
	}
	return mi
}

func iMinRedundancy(cay, cby, ca, cb, cy []float64, nBins int, total float64) float64 {
	if total <= 0.0 {
		return 0.0
	}
	iRed := 0.0
	for y := 0; y < nBins; y++ {
		if cy[y] <= 0.0 {
			continue
		}
		pY := cy[y] / total
		ispecA := 0.0
		for x := 0; x < nBins; x++ {
			if cay[x*nBins+y] <= 0.0 || ca[x] <= 0.0 {
				continue
			}
			pAGivenY := cay[x*nBins+y] / cy[y]
			pYGivenA := cay[x*nBins+y] / ca[x]
			ispecA += pAGivenY * math.Log(pYGivenA/pY)
		}
		ispecB := 0.0
		for x := 0; x < nBins; x++ {
			if cby[x*nBins+y] <= 0.0 || cb[x] <= 0.0 {
				continue
			}
			pBGivenY := cby[x*nBins+y] / cy[y]
			pYGivenB := cby[x*nBins+y] / cb[x]
			ispecB += pBGivenY * math.Log(pYGivenB/pY)
		}
		minSpec := ispecA
		if ispecB < minSpec {
			minSpec = ispecB
		}
		iRed += pY * minSpec
	}
	if iRed < 0.0 {
		return 0.0
	}
	return iRed
}

func pidDecomposition(
	history []float64, t, n int,
	groupA, groupB []int, nBins int,
) (float64, float64) {
	if t == 0 || n == 0 || len(groupA) == 0 || len(groupB) == 0 || nBins == 0 {
		return 0.0, 0.0
	}
	cy := make([]float64, nBins)
	ca := make([]float64, nBins)
	cb := make([]float64, nBins)
	cay := make([]float64, nBins*nBins)
	cby := make([]float64, nBins*nBins)
	cab := make([]float64, nBins*nBins)
	caby := make([]float64, nBins*nBins*nBins)

	for row := 0; row < t; row++ {
		y := binAngle(globalPhase(history, n, row), nBins)
		a := binAngle(groupPhase(history, n, row, groupA), nBins)
		b := binAngle(groupPhase(history, n, row, groupB), nBins)
		cy[y]++
		ca[a]++
		cb[b]++
		cay[a*nBins+y]++
		cby[b*nBins+y]++
		ab := a*nBins + b
		cab[ab]++
		caby[ab*nBins+y]++
	}

	total := float64(t)
	miA := mutualInformation(cay, ca, cy, nBins, nBins, total)
	miB := mutualInformation(cby, cb, cy, nBins, nBins, total)
	miAB := mutualInformation(caby, cab, cy, nBins*nBins, nBins, total)
	iRed := iMinRedundancy(cay, cby, ca, cb, cy, nBins, total)
	syn := miAB - miA - miB + iRed
	if syn < 0.0 {
		syn = 0.0
	}
	return iRed, syn
}

//export PidDecomposition
func PidDecomposition(
	historyPtr *C.double, t C.int, n C.int,
	groupAPtr *C.longlong, nA C.int,
	groupBPtr *C.longlong, nB C.int,
	nBins C.int,
	redPtr *C.double, synPtr *C.double,
) C.int {
	tt := int(t)
	nn := int(n)
	history := unsafe.Slice((*float64)(unsafe.Pointer(historyPtr)), tt*nn)
	groupA := make([]int, int(nA))
	if int(nA) > 0 {
		raw := unsafe.Slice((*int64)(unsafe.Pointer(groupAPtr)), int(nA))
		for i, v := range raw {
			groupA[i] = int(v)
		}
	}
	groupB := make([]int, int(nB))
	if int(nB) > 0 {
		raw := unsafe.Slice((*int64)(unsafe.Pointer(groupBPtr)), int(nB))
		for i, v := range raw {
			groupB[i] = int(v)
		}
	}
	red, syn := pidDecomposition(history, tt, nn, groupA, groupB, int(nBins))
	*redPtr = C.double(red)
	*synPtr = C.double(syn)
	return 0
}

func main() {}
