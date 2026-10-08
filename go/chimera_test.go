// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Original Go chimera numerical and ABI contracts

package main

import (
	"math"
	"testing"
	"unsafe"
)

func TestChimeraAnalyticGraphs(t *testing.T) {
	cases := []struct {
		name                  string
		phases, knm, expected []float64
	}{
		{"self residue", []float64{2.0}, []float64{1e-16}, []float64{0.0}},
		{"disconnected extreme", []float64{1e308, -1e308}, []float64{0, 0, 0, 0}, []float64{0, 0}},
		{"connected extreme", []float64{1e308, -1e308}, []float64{0, 1, 1, 0}, []float64{1, 1}},
		{"directed degree average", []float64{0, 0, math.Pi}, []float64{0, 1e-300, 1e300, -1, 0, 0, 1, 0, 0}, []float64{0, 0, 1}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			out := make([]float64, len(test.phases))
			localOrderParameter(test.phases, test.knm, len(test.phases), out)
			for i, expected := range test.expected {
				if math.Abs(out[i]-expected) > 1e-12 {
					t.Fatalf("R[%d]=%g; want %g", i, out[i], expected)
				}
			}
		})
	}
}

// invokeV2 preserves inferred cgo pointer types without importing C in a Go test
// file (which the Go toolchain forbids). Every nonnil pointer refers to live,
// full-sized Go storage for exactly the supplied extent throughout the call.
func invokeV2[T ~float64, L ~uint32 | ~uint64, N ~int32, R ~int32](
	fn func(*T, L, *T, L, N, *T, L) R,
	phases, knm []float64, n int, out []float64,
) R {
	var p, k, o *T
	if len(phases) > 0 {
		p = (*T)(unsafe.Pointer(&phases[0]))
	}
	if len(knm) > 0 {
		k = (*T)(unsafe.Pointer(&knm[0]))
	}
	if len(out) > 0 {
		o = (*T)(unsafe.Pointer(&out[0]))
	}
	return fn(p, L(len(phases)), k, L(len(knm)), N(n), o, L(len(out)))
}

func TestChimeraExportedBuffersAndRecovery(t *testing.T) {
	cases := []struct {
		phases, knm []float64
		n           int
		rc          int32
	}{
		{[]float64{0, 0}, []float64{0, 1, 1, 0}, 2, 0},
		{[]float64{math.NaN()}, []float64{0}, 1, 2},
		{[]float64{math.Inf(1)}, []float64{0}, 1, 2},
		{[]float64{0}, []float64{math.NaN()}, 1, 2},
		{[]float64{0}, []float64{math.Inf(-1)}, 1, 2},
		{[]float64{0}, []float64{1e-14}, 1, 2},
		{[]float64{2}, []float64{1e-16}, 1, 0},
	}
	for _, test := range cases {
		out := make([]float64, test.n)
		for i := range out {
			out[i] = -99
		}
		rc := invokeV2(LocalOrderParameterV2, test.phases, test.knm, test.n, out)
		if int32(rc) != test.rc {
			t.Fatalf("rc=%d; want %d", rc, test.rc)
		}
		if test.rc != 0 {
			for _, value := range out {
				if value != -99 {
					t.Fatal("refusal modified output")
				}
			}
		}
	}
	// Valid full buffers recover on the same original exported entry point.
	out := []float64{-99, -99}
	if invokeV2(LocalOrderParameterV2, []float64{0, 0}, []float64{0, 1, 1, 0}, 2, out) != 0 || out[0] != 1 || out[1] != 1 {
		t.Fatal("valid recovery did not produce coherent neighbours")
	}
}

func TestChimeraExportedMetadataBeforePointerAccess(t *testing.T) {
	if LocalOrderParameter(nil, nil, -1, nil) != 1 {
		t.Fatal("legacy negative count was admitted")
	}
	if LocalOrderParameter(nil, nil, 0, nil) != 0 || LocalOrderParameterV2(nil, 0, nil, 0, 0, nil, 0) != 0 {
		t.Fatal("valid empty identity was refused")
	}
	for _, rc := range []int32{
		int32(LocalOrderParameterV2(nil, 0, nil, 0, -1, nil, 0)),
		int32(LocalOrderParameterV2(nil, 1, nil, 1, 0, nil, 1)),
		int32(LocalOrderParameterV2(nil, 1, nil, 4, 2, nil, 2)),
		int32(LocalOrderParameterV2(nil, 2, nil, 3, 2, nil, 2)),
		int32(LocalOrderParameterV2(nil, 2, nil, 4, 2, nil, 1)),
		int32(LocalOrderParameterV2(nil, 2, nil, 4, 2, nil, 2)),
		int32(LocalOrderParameterV2(nil, 0, nil, 0, 2147483647, nil, 0)),
	} {
		if rc != 1 {
			t.Fatalf("invalid exported metadata returned %d; want 1", rc)
		}
	}
}
