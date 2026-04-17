// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — AttnRes coupling modulation (Go port)

// Package main builds ``libattnres.so`` — a C-shared library exported
// via cgo so the Python side can call it through ctypes. The Python
// bridge lives at ``src/scpn_phase_orchestrator/coupling/_attnres_go.py``.
//
// Build with::
//
//	go build -buildmode=c-shared -o libattnres.so attnres.go
package main

import "C"

import (
	"errors"
	"math"
	"unsafe"
)

// attnres is the pure-Go implementation. Returns a new flat `n*n`
// slice; the caller is responsible for freeing it via `FreeResult`.
func attnres(
	knm []float64,
	theta []float64,
	n int,
	blockSize int,
	temperature float64,
	lambda float64,
) ([]float64, error) {
	if len(knm) != n*n {
		return nil, errors.New("knm length does not match n*n")
	}
	if len(theta) != n {
		return nil, errors.New("theta length does not match n")
	}
	if blockSize < 1 {
		return nil, errors.New("blockSize must be >= 1")
	}
	if temperature <= 0.0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return nil, errors.New("temperature must be finite and > 0")
	}
	if lambda < 0.0 {
		return nil, errors.New("lambda must be >= 0")
	}

	out := make([]float64, n*n)
	if lambda == 0.0 {
		copy(out, knm)
		return out, nil
	}

	invT := 1.0 / temperature
	rowwise := make([]float64, n*n)
	logits := make([]float64, n)

	for i := 0; i < n; i++ {
		for j := range logits {
			logits[j] = math.Inf(-1)
		}
		lo := i - blockSize
		if lo < 0 {
			lo = 0
		}
		hi := i + blockSize + 1
		if hi > n {
			hi = n
		}
		anyUnmasked := false
		rowOff := i * n
		for j := lo; j < hi; j++ {
			if j == i || knm[rowOff+j] == 0.0 {
				continue
			}
			logits[j] = math.Cos(theta[j]-theta[i]) * invT
			anyUnmasked = true
		}

		if anyUnmasked {
			rowMax := math.Inf(-1)
			for _, x := range logits {
				if x > rowMax {
					rowMax = x
				}
			}
			denom := 0.0
			for j := 0; j < n; j++ {
				if !math.IsInf(logits[j], -1) {
					e := math.Exp(logits[j] - rowMax)
					logits[j] = e
					denom += e
				} else {
					logits[j] = 0.0
				}
			}
			if denom > 0.0 {
				invDenom := 1.0 / denom
				for j := range logits {
					logits[j] *= invDenom
				}
			}
		} else {
			for j := range logits {
				logits[j] = 0.0
			}
		}

		for j := 0; j < n; j++ {
			rowwise[rowOff+j] = knm[rowOff+j] * (1.0 + lambda*logits[j])
		}
	}

	// Symmetrise (R + Rᵀ) / 2.
	for i := 0; i < n; i++ {
		rowI := i * n
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			out[rowI+j] = 0.5 * (rowwise[rowI+j] + rowwise[j*n+i])
		}
	}
	return out, nil
}

//export AttnResModulate
//
// AttnResModulate is the exported C-ABI entry point.
//
// Parameters: input/output pointers to `n*n` / `n` f64 arrays.
// The caller allocates `outPtr` with the right size. Returns 0 on
// success, non-zero on validation failure. A NULL returned message
// pointer means the call succeeded.
func AttnResModulate(
	knmPtr *C.double,
	thetaPtr *C.double,
	n C.int,
	blockSize C.int,
	temperature C.double,
	lambda C.double,
	outPtr *C.double,
) C.int {
	nn := int(n)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	theta := unsafe.Slice((*float64)(unsafe.Pointer(thetaPtr)), nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn)

	result, err := attnres(
		knm, theta, nn, int(blockSize), float64(temperature), float64(lambda),
	)
	if err != nil {
		return 1
	}
	copy(out, result)
	return 0
}

func main() {}
