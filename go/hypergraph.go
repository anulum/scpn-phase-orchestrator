// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hypergraph k-body Kuramoto (Go port)

// Package main builds ``libhypergraph.so`` — generalised k-body
// Kuramoto stepper with an optional pairwise-K component. Mirrors
// ``spo-engine/src/hypergraph.rs`` including the
// ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` expansion used when
// ``alpha_flat`` is all zeros.
//
// Build with::
//
//	go build -buildmode=c-shared -o libhypergraph.so hypergraph.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiHg = 2.0 * math.Pi

func hypergraphStep(
	theta, omegas, knm, alpha []float64,
	n int,
	edgeNodes []int64,
	edgeOffsets []int64,
	edgeStrengths []float64,
	zeta, psi float64,
	sinTh, cosTh, deriv []float64,
) {
	for i := 0; i < n; i++ {
		sinTh[i] = math.Sin(theta[i])
		cosTh[i] = math.Cos(theta[i])
	}
	hasPairwise := len(knm) == n*n
	alphaZero := true
	for _, a := range alpha {
		if a != 0.0 {
			alphaZero = false
			break
		}
	}
	var zsPsi, zcPsi float64
	if zeta != 0.0 {
		zsPsi = zeta * math.Sin(psi)
		zcPsi = zeta * math.Cos(psi)
	}

	for i := 0; i < n; i++ {
		pw := 0.0
		if hasPairwise {
			offset := i * n
			ci := cosTh[i]
			si := sinTh[i]
			if alphaZero {
				for j := 0; j < n; j++ {
					pw += knm[offset+j] * (sinTh[j]*ci - cosTh[j]*si)
				}
			} else {
				for j := 0; j < n; j++ {
					pw += knm[offset+j] *
						math.Sin(theta[j]-theta[i]-alpha[offset+j])
				}
			}
		}
		deriv[i] = omegas[i] + pw
		if zeta != 0.0 {
			deriv[i] += zsPsi*cosTh[i] - zcPsi*sinTh[i]
		}
	}

	nEdges := len(edgeOffsets)
	for e := 0; e < nEdges; e++ {
		start := int(edgeOffsets[e])
		end := len(edgeNodes)
		if e+1 < nEdges {
			end = int(edgeOffsets[e+1])
		}
		k := end - start
		phaseSum := 0.0
		for p := start; p < end; p++ {
			phaseSum += theta[edgeNodes[p]]
		}
		sigma := edgeStrengths[e]
		for p := start; p < end; p++ {
			m := int(edgeNodes[p])
			deriv[m] += sigma * math.Sin(phaseSum-float64(k)*theta[m])
		}
	}
}

func hypergraphRun(
	phases, omegas, knm, alpha []float64,
	n int,
	edgeNodes []int64,
	edgeOffsets []int64,
	edgeStrengths []float64,
	zeta, psi, dt float64,
	nSteps int,
	out []float64,
) {
	copy(out, phases)
	sinTh := make([]float64, n)
	cosTh := make([]float64, n)
	deriv := make([]float64, n)
	for s := 0; s < nSteps; s++ {
		hypergraphStep(out, omegas, knm, alpha, n,
			edgeNodes, edgeOffsets, edgeStrengths,
			zeta, psi, sinTh, cosTh, deriv)
		for i := 0; i < n; i++ {
			v := math.Mod(out[i]+dt*deriv[i], twoPiHg)
			if v < 0 {
				v += twoPiHg
			}
			out[i] = v
		}
	}
}

//export HypergraphRun
func HypergraphRun(
	phasesPtr, omegasPtr *C.double,
	n C.int,
	edgeNodesPtr, edgeOffsetsPtr *C.longlong,
	edgeStrengthsPtr *C.double,
	nEdgeNodes C.int, nEdges C.int,
	knmPtr *C.double, knmLen C.int,
	alphaPtr *C.double, alphaLen C.int,
	zeta, psi, dt C.double,
	nSteps C.int,
	outPtr *C.double,
) C.int {
	nn := int(n)
	phases := unsafe.Slice((*float64)(unsafe.Pointer(phasesPtr)), nn)
	omegas := unsafe.Slice((*float64)(unsafe.Pointer(omegasPtr)), nn)
	en := unsafe.Slice((*int64)(unsafe.Pointer(edgeNodesPtr)), int(nEdgeNodes))
	eo := unsafe.Slice((*int64)(unsafe.Pointer(edgeOffsetsPtr)), int(nEdges))
	es := unsafe.Slice(
		(*float64)(unsafe.Pointer(edgeStrengthsPtr)), int(nEdges),
	)
	var knm []float64
	if int(knmLen) > 0 {
		knm = unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), int(knmLen))
	}
	var alpha []float64
	if int(alphaLen) > 0 {
		alpha = unsafe.Slice(
			(*float64)(unsafe.Pointer(alphaPtr)), int(alphaLen),
		)
	}
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), nn)
	hypergraphRun(phases, omegas, knm, alpha, nn, en, eo, es,
		float64(zeta), float64(psi), float64(dt),
		int(nSteps), out)
	return 0
}

func main() {}
