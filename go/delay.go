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
	"fmt"
	"math"
	"unsafe"
)

const tauDelay = 2.0 * math.Pi

// delayedKuramotoRun validates and integrates one independent finite batch.
// Zero steps return an unwrapped copy; positive steps use simultaneous delayed
// sources and canonical half-open projection. Refusal never mutates input slices.
func delayedKuramotoRun(
	phasesInit, omegas, knmFlat, alphaFlat []float64,
	n int, zeta, psi, dt float64, delaySteps, nSteps int,
) ([]float64, error) {
	maxInt := int(^uint(0) >> 1)
	if n <= 0 || n > maxInt/n || len(phasesInit) != n || len(omegas) != n || len(knmFlat) != n*n || len(alphaFlat) != n*n {
		return nil, fmt.Errorf("delay dimensions must match the supplied buffers")
	}
	if delaySteps < 0 || nSteps < 0 || dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, fmt.Errorf("invalid delay counts or timestep")
	}
	for _, values := range [][]float64{phasesInit, omegas, knmFlat, alphaFlat, {zeta, psi}} {
		for _, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("delay inputs must be finite")
			}
		}
	}
	p := make([]float64, n)
	copy(p, phasesInit)
	if nSteps == 0 {
		return p, nil
	}
	effectiveDelay := min(delaySteps, nSteps)
	if effectiveDelay == maxInt || n > maxInt/(effectiveDelay+1) {
		return nil, fmt.Errorf("delay history size overflow")
	}
	maxBuf := effectiveDelay + 1
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
			raw := thetaI + dt*dtheta
			if math.IsNaN(raw) || math.IsInf(raw, 0) {
				return nil, fmt.Errorf("delay step %d produced nonfinite phases", i)
			}
			newp[ii] = math.Mod(raw, tauDelay)
			if newp[ii] < 0 {
				newp[ii] += tauDelay
			}
			if newp[ii] >= tauDelay || newp[ii] == 0.0 {
				newp[ii] = 0.0
			}
		}
		p, newp = newp, p
	}
	return p, nil
}

// DelayedKuramotoRun writes final phases only after a successful complete batch.
// A nonzero status refuses invalid controls or nonfinite arithmetic without
// modifying the caller's phase or output buffers.
//
//export DelayedKuramotoRun
func DelayedKuramotoRun(
	phasesPtr, omegasPtr, knmPtr, alphaPtr *C.double,
	n C.int, zeta C.double, psi C.double, dt C.double,
	delaySteps C.int, nSteps C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	maximumElements := int(^uint(0)>>1) / int(unsafe.Sizeof(C.double(0)))
	if nn <= 0 || nn > maximumElements/nn || phasesPtr == nil || omegasPtr == nil || knmPtr == nil || alphaPtr == nil || outPtr == nil {
		return 1
	}
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	alpha := unsafe.Slice((*float64)(unsafe.Pointer(alphaPtr)), nn*nn)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	result, err := delayedKuramotoRun(
		phases, omegas, knm, alpha, nn,
		float64(zeta), float64(psi), float64(dt),
		int(delaySteps), int(nSteps),
	)
	if err != nil {
		return 1
	}
	copy(out, result)
	return 0
}

func main() {}
