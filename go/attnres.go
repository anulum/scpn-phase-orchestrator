// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — AttnRes coupling modulation (Go multi-head)

// Package main builds ``libattnres.so`` — a C-shared library that the
// Python side calls through ctypes. Implements the full multi-head
// AttnRes algorithm matching the Rust / NumPy / Julia references.
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

func attnres(
	knm []float64,
	theta []float64,
	wQ []float64,
	wK []float64,
	wV []float64,
	wO []float64,
	n int,
	nHeads int,
	blockSize int,
	temperature float64,
	lambda float64,
) ([]float64, error) {
	if len(knm) != n*n {
		return nil, errors.New("knm length mismatch")
	}
	if len(theta) != n {
		return nil, errors.New("theta length mismatch")
	}
	if nHeads < 1 {
		return nil, errors.New("nHeads must be >= 1")
	}
	if temperature <= 0.0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return nil, errors.New("temperature must be finite and > 0")
	}
	if lambda < 0.0 {
		return nil, errors.New("lambda must be >= 0")
	}
	if len(wQ) != len(wK) || len(wQ) != len(wV) {
		return nil, errors.New("W_K/W_V must match W_Q")
	}
	if len(wQ)%nHeads != 0 {
		return nil, errors.New("W_Q not divisible by nHeads")
	}
	perHead := len(wQ) / nHeads
	dHeadF := math.Sqrt(float64(perHead) / float64(nHeads))
	dHead := int(math.Round(dHeadF))
	if dHead*dHead*nHeads != perHead {
		return nil, errors.New("cannot infer d_head")
	}
	dModel := nHeads * dHead
	if len(wO) != nHeads*perHead {
		return nil, errors.New("W_O shape mismatch")
	}

	out := make([]float64, n*n)
	if lambda == 0.0 {
		copy(out, knm)
		return out, nil
	}

	// 1. Fourier-feature embedding.
	x := make([]float64, n*dModel)
	for i := 0; i < n; i++ {
		for h := 0; h < dModel/2; h++ {
			freq := float64(h + 1)
			x[i*dModel+2*h] = math.Cos(freq * theta[i])
			x[i*dModel+2*h+1] = math.Sin(freq * theta[i])
		}
	}

	// 2. Per-head Q, K, V.
	q := make([]float64, nHeads*n*dHead)
	k := make([]float64, nHeads*n*dHead)
	v := make([]float64, nHeads*n*dHead)
	for h := 0; h < nHeads; h++ {
		for i := 0; i < n; i++ {
			for e := 0; e < dHead; e++ {
				qs := 0.0
				ks := 0.0
				vs := 0.0
				for d := 0; d < dModel; d++ {
					xd := x[i*dModel+d]
					idx := h*dModel*dHead + d*dHead + e
					qs += xd * wQ[idx]
					ks += xd * wK[idx]
					vs += xd * wV[idx]
				}
				q[h*n*dHead+i*dHead+e] = qs
				k[h*n*dHead+i*dHead+e] = ks
				v[h*n*dHead+i*dHead+e] = vs
			}
		}
	}

	// 3. Attention softmax per head.
	invScale := 1.0 / (math.Sqrt(float64(dHead)) * temperature)
	attn := make([]float64, nHeads*n*n)
	rowLogits := make([]float64, n)
	for h := 0; h < nHeads; h++ {
		for i := 0; i < n; i++ {
			for jj := range rowLogits {
				rowLogits[jj] = math.Inf(-1)
			}
			anyUnmasked := false
			for j := 0; j < n; j++ {
				if i == j || knm[i*n+j] == 0.0 {
					continue
				}
				if blockSize >= 0 {
					diff := i - j
					if diff < 0 {
						diff = -diff
					}
					if diff > blockSize {
						continue
					}
				}
				dot := 0.0
				for e := 0; e < dHead; e++ {
					dot += q[h*n*dHead+i*dHead+e] *
						k[h*n*dHead+j*dHead+e]
				}
				rowLogits[j] = dot * invScale
				anyUnmasked = true
			}
			if !anyUnmasked {
				continue
			}
			rowMax := math.Inf(-1)
			for _, x := range rowLogits {
				if x > rowMax {
					rowMax = x
				}
			}
			denom := 0.0
			for j := 0; j < n; j++ {
				if !math.IsInf(rowLogits[j], -1) {
					e := math.Exp(rowLogits[j] - rowMax)
					rowLogits[j] = e
					denom += e
				} else {
					rowLogits[j] = 0.0
				}
			}
			if denom > 0.0 {
				invDenom := 1.0 / denom
				for j := 0; j < n; j++ {
					attn[h*n*n+i*n+j] = rowLogits[j] * invDenom
				}
			}
		}
	}

	// 4. heads · V, concat.
	concatWidth := nHeads * dHead
	concat := make([]float64, n*concatWidth)
	for h := 0; h < nHeads; h++ {
		for i := 0; i < n; i++ {
			for e := 0; e < dHead; e++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += attn[h*n*n+i*n+j] *
						v[h*n*dHead+j*dHead+e]
				}
				concat[i*concatWidth+h*dHead+e] = s
			}
		}
	}

	// 5. Output projection.
	o := make([]float64, n*dModel)
	for i := 0; i < n; i++ {
		for d := 0; d < dModel; d++ {
			s := 0.0
			for c := 0; c < concatWidth; c++ {
				s += concat[i*concatWidth+c] * wO[c*dModel+d]
			}
			o[i*dModel+d] = s
		}
	}

	// 6. Cosine similarity aggregation.
	oNorm := make([]float64, n)
	for i := 0; i < n; i++ {
		s := 0.0
		for d := 0; d < dModel; d++ {
			val := o[i*dModel+d]
			s += val * val
		}
		oNorm[i] = math.Sqrt(s) + 1e-12
	}
	aAgg := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j || knm[i*n+j] == 0.0 {
				continue
			}
			if blockSize >= 0 {
				diff := i - j
				if diff < 0 {
					diff = -diff
				}
				if diff > blockSize {
					continue
				}
			}
			dot := 0.0
			for d := 0; d < dModel; d++ {
				dot += o[i*dModel+d] * o[j*dModel+d]
			}
			cosSim := dot / (oNorm[i] * oNorm[j])
			aAgg[i*n+j] = 0.5 * (1.0 + cosSim)
		}
	}

	// 7. Modulation + symmetrise.
	rowwise := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			rowwise[i*n+j] = knm[i*n+j] * (1.0 + lambda*aAgg[i*n+j])
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			out[i*n+j] = 0.5 * (rowwise[i*n+j] + rowwise[j*n+i])
		}
	}
	return out, nil
}

//export AttnResModulate
//
// AttnResModulate is the exported C-ABI entry point.
func AttnResModulate(
	knmPtr *C.double,
	thetaPtr *C.double,
	wQPtr *C.double,
	wKPtr *C.double,
	wVPtr *C.double,
	wOPtr *C.double,
	n C.int,
	nHeads C.int,
	blockSize C.int,
	temperature C.double,
	lambda C.double,
	outPtr *C.double,
) C.int {
	nn := int(n)
	nH := int(nHeads)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	theta := unsafe.Slice((*float64)(unsafe.Pointer(thetaPtr)), nn)
	// W matrices are nHeads * dModel * dHead; infer dHead from dModel and
	// trust caller shape — Python side always passes d_model = 8.
	// We cannot statically know d_model here so we trust wQ length. The
	// Go kernel does the arithmetic on the flat slices directly.
	// To obtain the length we need wQ's true size; cgo slices require it.
	// The Python side guarantees d_model = 8 and d_head = d_model / nHeads.
	// Derive wLen = nHeads * d_model * (d_model / nHeads) = d_model ** 2.
	// For safety we pass a separate arg would be nicer, but we keep the
	// C ABI stable by recovering d_model via the wO length constraint:
	// wO is (nHeads * d_head, d_model) = d_model rows · d_model cols, so
	// wOLen = d_model**2. Pass wO first then wQ/wK/wV: all same length
	// when d_model = nHeads * d_head and d_head = d_model / nHeads.
	wLen := 64 // default d_model = 8 -> 8*8 = 64; caller guarantees
	// We re-derive wLen by scanning wQPtr for a reasonable bound: use
	// nHeads * d_model * d_head. With the Python-side contract that
	// d_model = 8, d_head = 8/nHeads, wLen = 8 * 8 = 64 regardless of
	// nHeads. This is a documented contract (PHASE_EMBED_DIM = 8).
	_ = wLen
	// Accept the convention: d_model = 8 → all W buffers are 64 f64.
	dModel := 8
	dHead := dModel / nH
	qLen := nH * dModel * dHead
	oLen := nH * dHead * dModel

	wQ := unsafe.Slice((*float64)(unsafe.Pointer(wQPtr)), qLen)
	wK := unsafe.Slice((*float64)(unsafe.Pointer(wKPtr)), qLen)
	wV := unsafe.Slice((*float64)(unsafe.Pointer(wVPtr)), qLen)
	wO := unsafe.Slice((*float64)(unsafe.Pointer(wOPtr)), oLen)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn)

	result, err := attnres(
		knm, theta, wQ, wK, wV, wO, nn, nH,
		int(blockSize), float64(temperature), float64(lambda),
	)
	if err != nil {
		return 1
	}
	copy(out, result)
	return 0
}

func main() {}
