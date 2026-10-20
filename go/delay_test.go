// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Native Go delayed integration contracts

package main

import (
	"math"
	"testing"
	"unsafe"
)

// TestDelaySavedHistory binds the native recurrence to independently saved states.
func TestDelaySavedHistory(t *testing.T) {
	for _, delay := range []int{0, 1, 3, 20} {
		p := []float64{0.4, 1.1}
		omega := []float64{1, -0.7}
		knm := []float64{0.1, 0.5, -0.2, -0.1}
		alpha := []float64{0.01, 0.17, -0.2, 0}
		want := append([]float64(nil), p...)
		history := [][]float64{}
		for step := 0; step < 8; step++ {
			history = append(history, append([]float64(nil), want...))
			past := want
			if step >= delay {
				past = history[step-delay]
			}
			next := make([]float64, 2)
			for i, phase := range want {
				force := omega[i] + 0.3*math.Sin(0.7-phase)
				for j := range past {
					force += knm[i*2+j] * math.Sin(past[j]-phase-alpha[i*2+j])
				}
				next[i] = phase + 0.01*force
			}
			want = next
		}
		got, err := delayedKuramotoRun(p, omega, knm, alpha, 2, 0.3, 0.7, 0.01, delay, 8)
		if err != nil {
			t.Fatal(err)
		}
		for i, expected := range want {
			if math.Abs(got[i]-expected) > 1e-13 {
				t.Fatalf("delay=%d phase[%d]=%g; want %g", delay, i, got[i], expected)
			}
		}
		if p[0] != 0.4 || p[1] != 1.1 {
			t.Fatal("native integration modified its input")
		}
	}
}

// invokeDelay keeps every C pointer backed by live storage of its declared extent.
func invokeDelay[T ~float64, N ~int32, R ~int32](
	fn func(*T, *T, *T, *T, N, T, T, T, N, N, *T) R,
	phases, omega, knm, alpha []float64, n int, zeta, psi, dt float64,
	delay, steps int, out []float64,
) R {
	pointer := func(values []float64) *T {
		if len(values) == 0 {
			return nil
		}
		return (*T)(unsafe.Pointer(&values[0]))
	}
	return fn(pointer(phases), pointer(omega), pointer(knm), pointer(alpha), N(n),
		T(zeta), T(psi), T(dt), N(delay), N(steps), pointer(out))
}

// TestDelayExportedProjectionAndRefusal exercises the actual C-shared entry point.
func TestDelayExportedProjectionAndRefusal(t *testing.T) {
	for _, phase := range []float64{-math.SmallestNonzeroFloat64, math.Copysign(0, -1), math.Nextafter(tauDelay, 0)} {
		out := []float64{-99}
		rc := invokeDelay(DelayedKuramotoRun, []float64{phase}, []float64{0}, []float64{0}, []float64{0}, 1, 0, 0, 0.01, 1, 1, out)
		want := phase
		if phase <= 0 {
			want = 0
		}
		if rc != 0 || out[0] != want || math.Signbit(out[0]) {
			t.Fatalf("endpoint %g returned rc=%d out=%v", phase, rc, out)
		}
	}
	for _, controls := range [][3]float64{{math.MaxFloat64, math.MaxFloat64, 0.01}, {math.NaN(), 0, 0.01}, {0, 0, 0}} {
		out := []float64{-99}
		rc := invokeDelay(DelayedKuramotoRun, []float64{0}, []float64{controls[0]}, []float64{0}, []float64{0}, 1, controls[1], math.Pi/2, controls[2], 1, 1, out)
		if rc == 0 || out[0] != -99 {
			t.Fatalf("refused request published output: rc=%d out=%v", rc, out)
		}
	}
	if DelayedKuramotoRun(nil, nil, nil, nil, 0, 0, 0, 0.01, 1, 1, nil) == 0 {
		t.Fatal("null buffers were accepted")
	}
	for _, n := range []int{-1, (1 << 31) - 1} {
		out := []float64{-99}
		// Invalid metadata must be refused before interpreting these live pointers.
		rc := invokeDelay(DelayedKuramotoRun, []float64{0}, []float64{0}, []float64{0}, []float64{0}, n, 0, 0, 0.01, 1, 1, out)
		if rc == 0 || out[0] != -99 {
			t.Fatalf("invalid matrix byte extent n=%d published output", n)
		}
	}
}

// TestDelayZeroAndInvalidInputs covers native validation before allocation.
func TestDelayZeroAndInvalidInputs(t *testing.T) {
	p := []float64{-1}
	got, err := delayedKuramotoRun(p, []float64{0}, []float64{0}, []float64{0}, 1, 0, 0, 0.01, int(^uint(0)>>1), 0)
	if err != nil || got[0] != -1 {
		t.Fatalf("zero batch returned %v, %v", got, err)
	}
	got[0] = 9
	if p[0] != -1 {
		t.Fatal("zero batch aliases input")
	}
	for _, size := range []int{0, -1, 2, int(^uint(0) >> 1)} {
		if _, err := delayedKuramotoRun(p, []float64{0}, []float64{0}, []float64{0}, size, 0, 0, 0.01, 1, 1); err == nil {
			t.Fatalf("invalid dimension %d was accepted", size)
		}
	}
	for _, counts := range [][2]int{{-1, 1}, {1, -1}} {
		if _, err := delayedKuramotoRun(p, []float64{0}, []float64{0}, []float64{0}, 1, 0, 0, 0.01, counts[0], counts[1]); err == nil {
			t.Fatalf("invalid counts %v were accepted", counts)
		}
	}
}
