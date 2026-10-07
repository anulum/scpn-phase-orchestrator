// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — AttnRes coupling modulation (Go multi-head)

// Package main builds “libattnres.so“ — a C-shared library that the
// Python side calls through ctypes. Implements SPO multi-head phase attention
// matching the Rust / NumPy / Julia coupling law. This is an oscillator
// adaptation, not the depth-residual operator from arXiv:2603.15031.
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
	if n < 0 || (n > 0 && n > int(^uint(0)>>1)/n) {
		return nil, errors.New("n*n is outside the integer domain")
	}
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
	if lambda < 0.0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		return nil, errors.New("lambda must be finite and >= 0")
	}
	if blockSize != -1 && blockSize < 1 {
		return nil, errors.New("blockSize must be -1 or positive")
	}
	for _, values := range [][]float64{knm, theta, wQ, wK, wV, wO} {
		for _, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, errors.New("inputs must contain only finite values")
			}
		}
	}
	for i := 0; i < n; i++ {
		if math.Abs(knm[i*n+i]) > 1e-12 {
			return nil, errors.New("knm diagonal must be zero")
		}
		for j := 0; j < n; j++ {
			if math.Abs(knm[i*n+j]-knm[j*n+i]) > 1e-12+1e-12*math.Abs(knm[j*n+i]) {
				return nil, errors.New("knm must be symmetric")
			}
		}
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
	if dHead < 1 || dHead*dHead*nHeads != perHead {
		return nil, errors.New("cannot infer d_head")
	}
	dModel := nHeads * dHead
	if dModel < 2 || dModel%2 != 0 {
		return nil, errors.New("d_model must be even and >= 2")
	}
	if len(wO) != nHeads*perHead {
		return nil, errors.New("W_O shape mismatch")
	}

	out := make([]float64, n*n)
	if n == 0 || lambda == 0.0 {
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
	for _, values := range [][]float64{q, k, v} {
		for _, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, errors.New("attention projections must be finite")
			}
		}
	}
	invScale := (1.0 / math.Sqrt(float64(dHead))) / temperature
	if math.IsNaN(invScale) || math.IsInf(invScale, 0) {
		return nil, errors.New("attention scale must be finite")
	}
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
				if math.IsNaN(rowLogits[j]) || math.IsInf(rowLogits[j], 0) {
					return nil, errors.New("attention logits must be finite")
				}
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
		if math.IsNaN(oNorm[i]) || math.IsInf(oNorm[i], 0) {
			return nil, errors.New("attention output norms must be finite")
		}
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
				dot += (o[i*dModel+d] / oNorm[i]) * (o[j*dModel+d] / oNorm[j])
			}
			cosSim := math.Max(-1.0, math.Min(1.0, dot))
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
			out[i*n+j] = 0.5*rowwise[i*n+j] + 0.5*rowwise[j*n+i]
			if math.IsNaN(out[i*n+j]) || math.IsInf(out[i*n+j], 0) {
				return nil, errors.New("modulated coupling must be finite")
			}
		}
	}
	return out, nil
}

// AttnResModulate preserves the legacy D=8 ABI. Callers must supply six
// correctly sized buffers and an N*N output buffer. Use V2 for other widths.
//
//export AttnResModulate
func AttnResModulate(
	knmPtr, thetaPtr, wQPtr, wKPtr, wVPtr, wOPtr *C.double,
	n, nHeads, blockSize C.int,
	temperature, lambda C.double,
	outPtr *C.double,
) C.int {
	return AttnResModulateV2(knmPtr, thetaPtr, wQPtr, wKPtr, wVPtr, wOPtr,
		n, nHeads, 8, blockSize, temperature, lambda, outPtr)
}

// AttnResModulateV2 carries the projection width explicitly. Each projection
// buffer contains D*D doubles; coupling/output contain N*N, and theta N.
// Buffers must remain valid for the call. Invalid dimensions are refused
// before constructing any unsafe slice; numerical failures leave output intact.
//
//export AttnResModulateV2
func AttnResModulateV2(
	knmPtr, thetaPtr, wQPtr, wKPtr, wVPtr, wOPtr *C.double,
	n, nHeads, dModel, blockSize C.int,
	temperature, lambda C.double,
	outPtr *C.double,
) C.int {
	nn, heads, width := int(n), int(nHeads), int(dModel)
	maxElements := int(^uint(0)>>1) / int(unsafe.Sizeof(float64(0)))
	if nn < 0 || heads < 1 || width < 2 || width%2 != 0 || width%heads != 0 ||
		width > maxElements/width ||
		(nn > 0 && (nn > maxElements/nn || nn > maxElements/width || heads > maxElements/(nn*nn))) {
		return 1
	}
	if wQPtr == nil || wKPtr == nil || wVPtr == nil || wOPtr == nil ||
		(nn > 0 && (knmPtr == nil || thetaPtr == nil || outPtr == nil)) {
		return 1
	}
	result, err := attnres(
		unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn),
		unsafe.Slice((*float64)(unsafe.Pointer(thetaPtr)), nn),
		unsafe.Slice((*float64)(unsafe.Pointer(wQPtr)), width*width),
		unsafe.Slice((*float64)(unsafe.Pointer(wKPtr)), width*width),
		unsafe.Slice((*float64)(unsafe.Pointer(wVPtr)), width*width),
		unsafe.Slice((*float64)(unsafe.Pointer(wOPtr)), width*width),
		nn, heads, int(blockSize), float64(temperature), float64(lambda),
	)
	if err != nil {
		return 1
	}
	copy(unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn*nn), result)
	return 0
}

func main() {}
