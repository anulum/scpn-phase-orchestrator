// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Recurrence matrix kernels (Go port)

// Package main builds ``librecurrence.so`` — a C-shared library
// exporting the single-trajectory recurrence matrix and
// two-trajectory cross-recurrence matrix.
//
// Build with::
//
//	go build -buildmode=c-shared -o librecurrence.so recurrence.go
package main

import "C"

import (
	"math"
	"unsafe"
)

func squaredDistance(
	a, b []float64, ia, ib, d int, angular bool,
) float64 {
	s := 0.0
	if angular {
		for k := 0; k < d; k++ {
			delta := a[ia*d+k] - b[ib*d+k]
			c := 2.0 * math.Sin(delta/2.0)
			s += c * c
		}
	} else {
		for k := 0; k < d; k++ {
			delta := a[ia*d+k] - b[ib*d+k]
			s += delta * delta
		}
	}
	return s
}

func fillRecurrence(
	a, b []float64, t, d int, epsilon float64, angular bool, out []uint8,
) {
	epsSq := epsilon * epsilon
	for i := 0; i < t; i++ {
		for j := 0; j < t; j++ {
			if squaredDistance(a, b, i, j, d, angular) <= epsSq {
				out[i*t+j] = 1
			} else {
				out[i*t+j] = 0
			}
		}
	}
}

//export RecurrenceMatrix
//
// RecurrenceMatrix writes t*t uint8 entries into outPtr (row-major).
func RecurrenceMatrix(
	trajPtr *C.double,
	t C.int,
	d C.int,
	epsilon C.double,
	angular C.int,
	outPtr *C.uchar,
) C.int {
	tt := int(t)
	dd := int(d)
	traj := unsafe.Slice((*float64)(unsafe.Pointer(trajPtr)), tt*dd)
	out := unsafe.Slice((*uint8)(unsafe.Pointer(outPtr)), tt*tt)
	fillRecurrence(traj, traj, tt, dd, float64(epsilon), angular != 0, out)
	return 0
}

//export CrossRecurrenceMatrix
//
// CrossRecurrenceMatrix writes t*t uint8 entries into outPtr.
func CrossRecurrenceMatrix(
	trajAPtr *C.double,
	trajBPtr *C.double,
	t C.int,
	d C.int,
	epsilon C.double,
	angular C.int,
	outPtr *C.uchar,
) C.int {
	tt := int(t)
	dd := int(d)
	a := unsafe.Slice((*float64)(unsafe.Pointer(trajAPtr)), tt*dd)
	b := unsafe.Slice((*float64)(unsafe.Pointer(trajBPtr)), tt*dd)
	out := unsafe.Slice((*uint8)(unsafe.Pointer(outPtr)), tt*tt)
	fillRecurrence(a, b, tt, dd, float64(epsilon), angular != 0, out)
	return 0
}

func main() {}
