// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase-amplitude coupling (Go port)

// Package main builds ``libpac.so`` — a C-shared library exporting
// the Tort 2010 modulation index and the (N, N) PAC matrix.
//
// Build with::
//
//	go build -buildmode=c-shared -o libpac.so pac.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPi = 2.0 * math.Pi

func modulationIndexCore(
	theta []float64, amp []float64, nBins int,
) float64 {
	if nBins < 2 || len(theta) == 0 || len(amp) == 0 {
		return 0.0
	}
	n := len(theta)
	if len(amp) < n {
		n = len(amp)
	}
	binWidth := twoPi / float64(nBins)
	meanAmp := make([]float64, nBins)
	counts := make([]int, nBins)
	for i := 0; i < n; i++ {
		wrapped := math.Mod(theta[i], twoPi)
		if wrapped < 0 {
			wrapped += twoPi
		}
		k := int(wrapped / binWidth)
		if k >= nBins {
			k = nBins - 1
		}
		meanAmp[k] += amp[i]
		counts[k]++
	}
	for k := 0; k < nBins; k++ {
		if counts[k] > 0 {
			meanAmp[k] /= float64(counts[k])
		}
	}
	total := 0.0
	for _, v := range meanAmp {
		total += v
	}
	if total <= 0.0 {
		return 0.0
	}
	logN := math.Log(float64(nBins))
	if logN < 1e-15 {
		return 0.0
	}
	kl := 0.0
	for k := 0; k < nBins; k++ {
		pk := meanAmp[k] / total
		if pk > 0.0 {
			kl += pk * math.Log(pk*float64(nBins))
		}
	}
	mi := kl / logN
	if mi < 0.0 {
		mi = 0.0
	} else if mi > 1.0 {
		mi = 1.0
	}
	return mi
}

//export ModulationIndex
//
// ModulationIndex writes the Tort 2010 PAC MI into outMI. Returns 0
// on success; the kernel itself never errors on valid f64 inputs.
func ModulationIndex(
	thetaPtr *C.double,
	ampPtr *C.double,
	n C.int,
	nBins C.int,
	outMI *C.double,
) C.int {
	theta := unsafe.Slice((*float64)(unsafe.Pointer(thetaPtr)), int(n))
	amp := unsafe.Slice((*float64)(unsafe.Pointer(ampPtr)), int(n))
	*outMI = C.double(modulationIndexCore(theta, amp, int(nBins)))
	return 0
}

//export PACMatrix
//
// PACMatrix writes the flat (N, N) PAC matrix into outPtr (expected
// capacity n*n). phasesPtr and amplitudesPtr are flat (t, n) row-
// major buffers.
func PACMatrix(
	phasesPtr *C.double,
	amplitudesPtr *C.double,
	t C.int,
	n C.int,
	nBins C.int,
	outPtr *C.double,
) C.int {
	tt := int(t)
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), tt*nn)
	amps := unsafe.Slice((*float64)(unsafe.Pointer(amplitudesPtr)), tt*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn)

	thetaCol := make([]float64, tt)
	ampCol := make([]float64, tt)
	for i := 0; i < nn; i++ {
		for s := 0; s < tt; s++ {
			thetaCol[s] = phases[s*nn+i]
		}
		for j := 0; j < nn; j++ {
			for s := 0; s < tt; s++ {
				ampCol[s] = amps[s*nn+j]
			}
			out[i*nn+j] = modulationIndexCore(thetaCol, ampCol, int(nBins))
		}
	}
	return 0
}

func main() {}
