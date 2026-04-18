// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Lyapunov spectrum (Go port)

// Package main builds ``liblyapunov.so`` — a C-shared library exporting
// the Benettin 1980 / Shimada-Nagashima 1979 Lyapunov spectrum kernel
// on the Kuramoto tangent space with RK4 integration and periodic
// row-oriented Modified Gram-Schmidt. Matches the NumPy, Rust, Julia,
// and Mojo reference implementations bit-for-bit up to float rounding.
//
// Build with::
//
//	go build -buildmode=c-shared -o liblyapunov.so lyapunov.go
package main

import "C"

import (
	"math"
	"sort"
	"unsafe"
)

const twoPi = 2.0 * math.Pi

// kuramotoRHS evaluates dθ/dt = ω + coupling + driving at the given
// phases and writes the result into out (length n).
func kuramotoRHS(
	phases, omegas, knm, alpha []float64,
	n int, zeta, psi float64,
	out []float64,
) {
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += knm[i*n+j] * math.Sin(phases[j]-phases[i]-alpha[i*n+j])
		}
		driving := 0.0
		if zeta != 0.0 {
			driving = zeta * math.Sin(psi-phases[i])
		}
		out[i] = omegas[i] + s + driving
	}
}

// kuramotoJacobian fills J (row-major n×n) with the Jacobian of the
// Kuramoto RHS at the given phases. Diagonal includes the driver term
// −ζ cos(Ψ − θ_i).
func kuramotoJacobian(
	phases, knm, alpha []float64,
	n int, zeta, psi float64,
	J []float64,
) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				J[i*n+j] = 0.0
				continue
			}
			J[i*n+j] = knm[i*n+j] * math.Cos(phases[j]-phases[i]-alpha[i*n+j])
		}
	}
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			if i != j {
				s += J[i*n+j]
			}
		}
		driverDiag := 0.0
		if zeta != 0.0 {
			driverDiag = zeta * math.Cos(psi-phases[i])
		}
		J[i*n+i] = -(s + driverDiag)
	}
}

// matMul writes out = A · B (row-major n×n).
func matMul(A, B, out []float64, n int) {
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			s := 0.0
			for j := 0; j < n; j++ {
				s += A[i*n+j] * B[j*n+k]
			}
			out[i*n+k] = s
		}
	}
}

// rowMGS performs Modified Gram-Schmidt on rows of Q (row-major n×n) in
// place. After the call, each row of Q is orthonormal and diagR[k] is
// |R_kk|. Two-pass reorthogonalisation (Daniel et al. 1976) for
// numerical stability, matching the Rust kernel.
func rowMGS(Q []float64, n int, diagR []float64) {
	for j := 0; j < n; j++ {
		for pass := 0; pass < 2; pass++ {
			for k := 0; k < j; k++ {
				dot := 0.0
				for i := 0; i < n; i++ {
					dot += Q[k*n+i] * Q[j*n+i]
				}
				for i := 0; i < n; i++ {
					Q[j*n+i] -= dot * Q[k*n+i]
				}
			}
		}
		normSq := 0.0
		for i := 0; i < n; i++ {
			v := Q[j*n+i]
			normSq += v * v
		}
		norm := math.Sqrt(normSq)
		diagR[j] = norm
		if norm > 1e-300 {
			inv := 1.0 / norm
			for i := 0; i < n; i++ {
				Q[j*n+i] *= inv
			}
		}
	}
}

func lyapunovSpectrum(
	phasesInit, omegas, knm, alpha []float64,
	n int,
	dt float64,
	nSteps, qrInterval int,
	zeta, psi float64,
) []float64 {
	phases := make([]float64, n)
	copy(phases, phasesInit)

	nn := n * n
	Q := make([]float64, nn)
	for i := 0; i < n; i++ {
		Q[i*n+i] = 1.0
	}

	// RK4 stage buffers.
	k1p := make([]float64, n)
	k2p := make([]float64, n)
	k3p := make([]float64, n)
	k4p := make([]float64, n)
	k1q := make([]float64, nn)
	k2q := make([]float64, nn)
	k3q := make([]float64, nn)
	k4q := make([]float64, nn)
	tmpP := make([]float64, n)
	tmpQ := make([]float64, nn)
	J := make([]float64, nn)
	diagR := make([]float64, n)
	exponents := make([]float64, n)

	totalTime := 0.0
	dt6 := dt / 6.0

	for step := 0; step < nSteps; step++ {
		// --- Stage 1 -------------------------------------------------
		kuramotoRHS(phases, omegas, knm, alpha, n, zeta, psi, k1p)
		kuramotoJacobian(phases, knm, alpha, n, zeta, psi, J)
		matMul(J, Q, k1q, n)

		// --- Stage 2 -------------------------------------------------
		for i := 0; i < n; i++ {
			tmpP[i] = phases[i] + 0.5*dt*k1p[i]
		}
		for i := 0; i < nn; i++ {
			tmpQ[i] = Q[i] + 0.5*dt*k1q[i]
		}
		kuramotoRHS(tmpP, omegas, knm, alpha, n, zeta, psi, k2p)
		kuramotoJacobian(tmpP, knm, alpha, n, zeta, psi, J)
		matMul(J, tmpQ, k2q, n)

		// --- Stage 3 -------------------------------------------------
		for i := 0; i < n; i++ {
			tmpP[i] = phases[i] + 0.5*dt*k2p[i]
		}
		for i := 0; i < nn; i++ {
			tmpQ[i] = Q[i] + 0.5*dt*k2q[i]
		}
		kuramotoRHS(tmpP, omegas, knm, alpha, n, zeta, psi, k3p)
		kuramotoJacobian(tmpP, knm, alpha, n, zeta, psi, J)
		matMul(J, tmpQ, k3q, n)

		// --- Stage 4 -------------------------------------------------
		for i := 0; i < n; i++ {
			tmpP[i] = phases[i] + dt*k3p[i]
		}
		for i := 0; i < nn; i++ {
			tmpQ[i] = Q[i] + dt*k3q[i]
		}
		kuramotoRHS(tmpP, omegas, knm, alpha, n, zeta, psi, k4p)
		kuramotoJacobian(tmpP, knm, alpha, n, zeta, psi, J)
		matMul(J, tmpQ, k4q, n)

		// --- Combine + wrap -----------------------------------------
		for i := 0; i < n; i++ {
			next := phases[i] + dt6*(k1p[i]+2.0*k2p[i]+2.0*k3p[i]+k4p[i])
			phases[i] = math.Mod(next, twoPi)
			if phases[i] < 0.0 {
				phases[i] += twoPi
			}
		}
		for i := 0; i < nn; i++ {
			Q[i] += dt6 * (k1q[i] + 2.0*k2q[i] + 2.0*k3q[i] + k4q[i])
		}
		totalTime += dt

		// --- Periodic QR reorthogonalisation -------------------------
		if (step+1)%qrInterval == 0 {
			rowMGS(Q, n, diagR)
			for i := 0; i < n; i++ {
				d := math.Abs(diagR[i])
				if d < 1e-300 {
					d = 1e-300
				}
				exponents[i] += math.Log(d)
			}
		}
	}

	if totalTime > 0.0 {
		for i := 0; i < n; i++ {
			exponents[i] /= totalTime
		}
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(exponents)))
	return exponents
}

//export LyapunovSpectrum
//
// LyapunovSpectrum writes n exponents (sorted descending) into outPtr.
// knmFlat and alphaFlat are row-major n×n matrices; outPtr expects n
// cells. Returns 0 on success.
func LyapunovSpectrum(
	phasesInitPtr *C.double,
	omegasPtr *C.double,
	knmFlatPtr *C.double,
	alphaFlatPtr *C.double,
	n C.int,
	dt C.double,
	nSteps C.int,
	qrInterval C.int,
	zeta C.double,
	psi C.double,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesInitPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmFlatPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaFlatPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)

	result := lyapunovSpectrum(
		phases, omegas, knm, alpha, nn,
		float64(dt), int(nSteps), int(qrInterval),
		float64(zeta), float64(psi),
	)
	copy(out, result)
	return 0
}

func main() {}
