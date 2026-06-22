// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Koopman EDMD-with-control solve (Go port)

// Package main builds ``libkoopman_edmd.so`` — a C-shared library exporting the
// Koopman EDMD-with-control least-squares solve (Korda & Mezić 2018). The
// algorithm matches the Rust, NumPy, Julia, and Mojo reference implementations
// to machine precision.
//
// Build with::
//
//	go build -buildmode=c-shared -o libkoopman_edmd.so koopman_edmd.go
package main

import "C"

import "unsafe"

// solveMulti solves ``mat · X = rhs`` for X (dim×nRhs) by Gaussian elimination
// with partial pivoting; mat is row-major (dim×dim), rhs row-major (dim×nRhs).
func solveMulti(dim int, mat []float64, rhs []float64, nRhs int) ([]float64, bool) {
	width := dim + nRhs
	aug := make([]float64, dim*width)
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			aug[r*width+c] = mat[r*dim+c]
		}
		for c := 0; c < nRhs; c++ {
			aug[r*width+dim+c] = rhs[r*nRhs+c]
		}
	}
	for col := 0; col < dim; col++ {
		pivotRow := col
		pivotMag := abs(aug[col*width+col])
		for r := col + 1; r < dim; r++ {
			mag := abs(aug[r*width+col])
			if mag > pivotMag {
				pivotMag = mag
				pivotRow = r
			}
		}
		if pivotMag == 0.0 {
			return nil, false
		}
		if pivotRow != col {
			for c := 0; c < width; c++ {
				aug[col*width+c], aug[pivotRow*width+c] = aug[pivotRow*width+c], aug[col*width+c]
			}
		}
		pivot := aug[col*width+col]
		for r := 0; r < dim; r++ {
			if r == col {
				continue
			}
			factor := aug[r*width+col] / pivot
			if factor == 0.0 {
				continue
			}
			for c := col; c < width; c++ {
				aug[r*width+c] -= factor * aug[col*width+c]
			}
		}
	}
	sol := make([]float64, dim*nRhs)
	for r := 0; r < dim; r++ {
		pivot := aug[r*width+r]
		for c := 0; c < nRhs; c++ {
			sol[r*nRhs+c] = aug[r*width+dim+c] / pivot
		}
	}
	return sol, true
}

func abs(value float64) float64 {
	if value < 0.0 {
		return -value
	}
	return value
}

//export KoopmanEdmdSolve
func KoopmanEdmdSolve(
	xLiftPtr *C.double, inputsPtr *C.double, yLiftPtr *C.double, statesPtr *C.double,
	kArg, nLiftArg, mArg, nStateArg C.int, reg C.double,
	aOut *C.double, bOut *C.double, cOut *C.double,
) C.int {
	k := int(kArg)
	nLift := int(nLiftArg)
	m := int(mArg)
	nState := int(nStateArg)
	if k <= 0 || nLift <= 0 || nState <= 0 {
		return 1
	}
	xLift := unsafe.Slice((*float64)(unsafe.Pointer(xLiftPtr)), k*nLift)
	inputs := unsafe.Slice((*float64)(unsafe.Pointer(inputsPtr)), k*m)
	yLift := unsafe.Slice((*float64)(unsafe.Pointer(yLiftPtr)), k*nLift)
	states := unsafe.Slice((*float64)(unsafe.Pointer(statesPtr)), k*nState)

	p := nLift + m
	gram := make([]float64, p*p)
	cross := make([]float64, p*nLift)
	for i := 0; i < k; i++ {
		for a := 0; a < p; a++ {
			var phiA float64
			if a < nLift {
				phiA = xLift[i*nLift+a]
			} else {
				phiA = inputs[i*m+(a-nLift)]
			}
			for b := 0; b < p; b++ {
				var phiB float64
				if b < nLift {
					phiB = xLift[i*nLift+b]
				} else {
					phiB = inputs[i*m+(b-nLift)]
				}
				gram[a*p+b] += phiA * phiB
			}
			for j := 0; j < nLift; j++ {
				cross[a*nLift+j] += phiA * yLift[i*nLift+j]
			}
		}
	}
	regular := float64(reg)
	for a := 0; a < p; a++ {
		gram[a*p+a] += regular
	}
	mSol, ok := solveMulti(p, gram, cross, nLift)
	if !ok {
		return 2
	}

	aSlice := unsafe.Slice((*float64)(unsafe.Pointer(aOut)), nLift*nLift)
	bSlice := unsafe.Slice((*float64)(unsafe.Pointer(bOut)), nLift*m)
	cSlice := unsafe.Slice((*float64)(unsafe.Pointer(cOut)), nState*nLift)
	for i := 0; i < nLift; i++ {
		for c := 0; c < nLift; c++ {
			aSlice[i*nLift+c] = mSol[c*nLift+i]
		}
		for c := 0; c < m; c++ {
			bSlice[i*m+c] = mSol[(nLift+c)*nLift+i]
		}
	}

	liftGram := make([]float64, nLift*nLift)
	cc := make([]float64, nLift*nState)
	for i := 0; i < k; i++ {
		for a := 0; a < nLift; a++ {
			xa := xLift[i*nLift+a]
			for b := 0; b < nLift; b++ {
				liftGram[a*nLift+b] += xa * xLift[i*nLift+b]
			}
			for j := 0; j < nState; j++ {
				cc[a*nState+j] += xa * states[i*nState+j]
			}
		}
	}
	for a := 0; a < nLift; a++ {
		liftGram[a*nLift+a] += regular
	}
	ct, ok := solveMulti(nLift, liftGram, cc, nState)
	if !ok {
		return 3
	}
	for i := 0; i < nState; i++ {
		for c := 0; c < nLift; c++ {
			cSlice[i*nLift+c] = ct[c*nState+i]
		}
	}
	return 0
}

func main() {}
