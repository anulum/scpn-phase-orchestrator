// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ott-Antonsen reduction (Go port)

// Package main builds ``libreduction.so`` — RK4 integrator for
// the Ott-Antonsen complex-scalar mean-field ODE. Mirrors
// ``spo-engine/src/reduction.rs`` bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libreduction.so reduction.go
package main

import "C"

import (
	"math"
)

func oaDeriv(re, im, omega0, delta, halfK float64) (float64, float64) {
	absSq := re*re + im*im
	linRe := -delta*re + omega0*im
	linIm := -delta*im - omega0*re
	cubicFactor := halfK * (1.0 - absSq)
	cubRe := cubicFactor * re
	cubIm := cubicFactor * im
	return linRe + cubRe, linIm + cubIm
}

func oaRun(
	zRe, zIm, omega0, delta, kCoupling, dt float64,
	nSteps int,
) (float64, float64, float64, float64) {
	re := zRe
	im := zIm
	halfK := kCoupling / 2.0
	for s := 0; s < nSteps; s++ {
		k1r, k1i := oaDeriv(re, im, omega0, delta, halfK)
		k2r, k2i := oaDeriv(re+0.5*dt*k1r, im+0.5*dt*k1i,
			omega0, delta, halfK)
		k3r, k3i := oaDeriv(re+0.5*dt*k2r, im+0.5*dt*k2i,
			omega0, delta, halfK)
		k4r, k4i := oaDeriv(re+dt*k3r, im+dt*k3i,
			omega0, delta, halfK)
		re += (dt / 6.0) * (k1r + 2.0*k2r + 2.0*k3r + k4r)
		im += (dt / 6.0) * (k1i + 2.0*k2i + 2.0*k3i + k4i)
	}
	r := math.Sqrt(re*re + im*im)
	psi := math.Atan2(im, re)
	return re, im, r, psi
}

//export OARun
func OARun(
	zRe, zIm, omega0, delta, kCoupling, dt C.double,
	nSteps C.int,
	outRe, outIm, outR, outPsi *C.double,
) C.int {
	re, im, r, psi := oaRun(
		float64(zRe), float64(zIm), float64(omega0),
		float64(delta), float64(kCoupling), float64(dt),
		int(nSteps),
	)
	*outRe = C.double(re)
	*outIm = C.double(im)
	*outR = C.double(r)
	*outPsi = C.double(psi)
	return 0
}

func main() {}
