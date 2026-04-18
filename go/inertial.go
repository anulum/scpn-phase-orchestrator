// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Second-order inertial Kuramoto (Go port)

// Package main builds ``libinertial.so`` — second-order swing-equation
// Kuramoto RK4 stepper. The derivative uses the Rust kernel's sincos
// expansion ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` so the Go output
// matches Rust bit-for-bit.
//
// Build with::
//
//	go build -buildmode=c-shared -o libinertial.so inertial.go
package main

import "C"

import (
	"math"
	"unsafe"
)

const twoPiInertial = 2.0 * math.Pi

func computeDerivative(
	theta, omegaDot, power, knm, inertia, damping []float64,
	n int,
	outT, outO, sinTh, cosTh []float64,
) {
	for i := 0; i < n; i++ {
		sinTh[i] = math.Sin(theta[i])
		cosTh[i] = math.Cos(theta[i])
	}
	for i := 0; i < n; i++ {
		outT[i] = omegaDot[i]
		ci := cosTh[i]
		si := sinTh[i]
		offset := i * n
		coupling := 0.0
		for j := 0; j < n; j++ {
			coupling += knm[offset+j] * (sinTh[j]*ci - cosTh[j]*si)
		}
		outO[i] = (power[i] + coupling - damping[i]*omegaDot[i]) / inertia[i]
	}
}

func inertialStep(
	theta, omegaDot, power, knm, inertia, damping []float64,
	n int,
	dt float64,
	newTheta, newOmegaDot []float64,
) {
	k1t := make([]float64, n)
	k1o := make([]float64, n)
	k2t := make([]float64, n)
	k2o := make([]float64, n)
	k3t := make([]float64, n)
	k3o := make([]float64, n)
	k4t := make([]float64, n)
	k4o := make([]float64, n)
	tmpTh := make([]float64, n)
	tmpOd := make([]float64, n)
	sinTh := make([]float64, n)
	cosTh := make([]float64, n)

	computeDerivative(theta, omegaDot, power, knm, inertia, damping, n,
		k1t, k1o, sinTh, cosTh)
	for i := 0; i < n; i++ {
		tmpTh[i] = theta[i] + 0.5*dt*k1t[i]
		tmpOd[i] = omegaDot[i] + 0.5*dt*k1o[i]
	}
	computeDerivative(tmpTh, tmpOd, power, knm, inertia, damping, n,
		k2t, k2o, sinTh, cosTh)
	for i := 0; i < n; i++ {
		tmpTh[i] = theta[i] + 0.5*dt*k2t[i]
		tmpOd[i] = omegaDot[i] + 0.5*dt*k2o[i]
	}
	computeDerivative(tmpTh, tmpOd, power, knm, inertia, damping, n,
		k3t, k3o, sinTh, cosTh)
	for i := 0; i < n; i++ {
		tmpTh[i] = theta[i] + dt*k3t[i]
		tmpOd[i] = omegaDot[i] + dt*k3o[i]
	}
	computeDerivative(tmpTh, tmpOd, power, knm, inertia, damping, n,
		k4t, k4o, sinTh, cosTh)

	dt6 := dt / 6.0
	for i := 0; i < n; i++ {
		raw := theta[i] + dt6*(k1t[i]+2.0*k2t[i]+2.0*k3t[i]+k4t[i])
		v := math.Mod(raw, twoPiInertial)
		if v < 0 {
			v += twoPiInertial
		}
		newTheta[i] = v
		newOmegaDot[i] = omegaDot[i] +
			dt6*(k1o[i]+2.0*k2o[i]+2.0*k3o[i]+k4o[i])
	}
}

//export InertialStep
func InertialStep(
	thetaPtr, omegaDotPtr, powerPtr, knmPtr,
	inertiaPtr, dampingPtr *C.double,
	n C.int,
	dt C.double,
	newThetaPtr, newOmegaDotPtr *C.double,
) C.int {
	nn := int(n)
	theta := unsafe.Slice((*float64)(unsafe.Pointer(thetaPtr)), nn)
	omegaDot := unsafe.Slice((*float64)(unsafe.Pointer(omegaDotPtr)), nn)
	power := unsafe.Slice((*float64)(unsafe.Pointer(powerPtr)), nn)
	knm := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), nn*nn)
	inertia := unsafe.Slice((*float64)(unsafe.Pointer(inertiaPtr)), nn)
	damping := unsafe.Slice((*float64)(unsafe.Pointer(dampingPtr)), nn)
	newTheta := unsafe.Slice((*float64)(unsafe.Pointer(newThetaPtr)), nn)
	newOmegaDot := unsafe.Slice(
		(*float64)(unsafe.Pointer(newOmegaDotPtr)), nn,
	)
	inertialStep(theta, omegaDot, power, knm, inertia, damping, nn,
		float64(dt), newTheta, newOmegaDot)
	return 0
}

func main() {}
