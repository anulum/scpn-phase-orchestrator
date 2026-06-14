// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Time-delayed Kuramoto integration (Go port)

// Package main builds libdelay.so — explicit-Euler integration of the
// time-delayed Kuramoto model
//
//	dθ_i/dt = ω_i + Σ_j K_ij·sin(θ_j(t−τ) − θ_i − α_ij) + ζ·sin(Ψ − θ_i)
//
// A ring buffer of delay_steps+1 phase snapshots supplies the delayed source
// phase; the first delay_steps steps use the current snapshot (zero-delay
// warmup), matching the NumPy reference.
//
// Build with::
//
//	go build -buildmode=c-shared -o libdelay.so delay.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const tauDelay = 2.0 * math.Pi

func delayedKuramotoRun(
	phasesInit, omegas, knmFlat, alphaFlat []float64,
	n int, zeta, psi, dt float64, delaySteps, nSteps int,
) []float64 {
	maxBuf := delaySteps + 1
	p := make([]float64, n)
	copy(p, phasesInit)
	newp := make([]float64, n)
	hist := make([]float64, maxBuf*n)

	alphaZero := true
	for _, a := range alphaFlat {
		if a != 0.0 {
			alphaZero = false
			break
		}
	}

	for i := 0; i < nSteps; i++ {
		ring := i % maxBuf
		for j := 0; j < n; j++ {
			hist[ring*n+j] = p[j]
		}
		didx := ring
		if delaySteps > 0 && i >= delaySteps {
			didx = (i - delaySteps) % maxBuf
		}
		for ii := 0; ii < n; ii++ {
			thetaI := p[ii]
			row := ii * n
			coupling := 0.0
			for jj := 0; jj < n; jj++ {
				dj := hist[didx*n+jj]
				if alphaZero {
					coupling += knmFlat[row+jj] * math.Sin(dj-thetaI)
				} else {
					coupling += knmFlat[row+jj] * math.Sin(dj-thetaI-alphaFlat[row+jj])
				}
			}
			dtheta := omegas[ii] + coupling
			if zeta != 0.0 {
				dtheta += zeta * math.Sin(psi-thetaI)
			}
			newp[ii] = math.Mod(thetaI+dt*dtheta, tauDelay)
			if newp[ii] < 0 {
				newp[ii] += tauDelay
			}
		}
		p, newp = newp, p
	}
	return p
}

//export DelayedKuramotoRun
func DelayedKuramotoRun(
	phasesPtr, omegasPtr, knmPtr, alphaPtr *C.double,
	n C.int, zeta C.double, psi C.double, dt C.double,
	delaySteps C.int, nSteps C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	result := delayedKuramotoRun(
		phases, omegas, knm, alpha, nn,
		float64(zeta), float64(psi), float64(dt),
		int(delaySteps), int(nSteps),
	)
	copy(out, result)
	return 0
}

func main() {}
