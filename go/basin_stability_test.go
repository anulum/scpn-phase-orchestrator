// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Native deterministic trial contracts

package main

import (
	"math"
	"testing"
	"unsafe"
)

func TestTinyEdgeAndPostStepWindows(t *testing.T) {
	p, o, k, a := []float64{0, math.Pi / 2}, make([]float64, 2), []float64{0, 5e-31, 5e-31, 0}, make([]float64, 4)
	expected := math.Cos((math.Pi/2 - 1) / 2)
	if value := steadyStateR(p, o, k, a, 2, 1, 1e30, 0, 1); math.Abs(value-expected) > 2e-15 {
		t.Fatalf("tiny edge %.17g != %.17g", value, expected)
	}
	if p[0] != 0 || p[1] != math.Pi/2 {
		t.Fatal("caller phases changed")
	}
	if value := steadyStateR([]float64{0, 1}, []float64{1e308, 1e308}, a, a, 2, 1, 2, 100, 0); value != 0 {
		t.Fatal("empty window integrated")
	}
	if value := steadyStateR([]float64{1e308, -1e308}, o, a, a, 2, 1, 1, 0, 1); math.Abs(value-math.Abs(math.Cos(1e308))) > 2e-15 {
		t.Fatal("zero edges evaluated overflowing differences")
	}
}

func TestDirectedLaggedAndSelfEdges(t *testing.T) {
	p, o, k, a := []float64{0, math.Pi / 2}, []float64{0.2, -0.1}, []float64{0.4, -0.7, 0, 0.5}, []float64{0.3, -0.2, 0, -0.4}
	theta0 := 0.09 * (o[0] + k[0]*math.Sin(-a[0]) + k[1]*math.Sin(math.Pi/2-a[1]))
	theta1 := math.Pi/2 + 0.09*(o[1]+k[3]*math.Sin(-a[3]))
	expected := math.Abs(math.Cos((theta1 - theta0) / 2))
	if value := steadyStateR(p, o, k, a, 2, 1, 0.09, 0, 1); math.Abs(value-expected) > 2e-15 {
		t.Fatalf("directed edge result %.17g", value)
	}
}

func TestInvalidDomainsAndArithmetic(t *testing.T) {
	p, o, z := []float64{0, 1}, make([]float64, 2), make([]float64, 4)
	cases := []struct {
		p, o, k, a         []float64
		n                  int
		scale, dt          float64
		transient, measure int
	}{
		{p, o, z, z, 0, 1, 0.1, 0, 1}, {p, o, z, z, 2, 1, 0.1, -1, 1}, {p, o, z, z, 2, 1, 0.1, 0, -1},
		{p[:1], o, z, z, 2, 1, 0.1, 0, 1}, {p, o[:1], z, z, 2, 1, 0.1, 0, 1}, {p, o, z[:3], z, 2, 1, 0.1, 0, 1}, {p, o, z, z[:3], 2, 1, 0.1, 0, 1},
		{p, o, z, z, 2, math.Inf(1), 0.1, 0, 1}, {p, o, z, z, 2, 1, 0, 0, 1}, {p, o, z, z, 2, 1, math.NaN(), 0, 1},
		{[]float64{math.NaN(), 0}, o, z, z, 2, 1, 0.1, 0, 0}, {p, []float64{math.Inf(1), 0}, z, z, 2, 1, 0.1, 0, 0},
		{p, o, []float64{0, math.NaN(), 0, 0}, z, 2, 1, 0.1, 0, 0}, {p, o, z, []float64{0, math.Inf(1), 0, 0}, 2, 1, 0.1, 0, 0},
		{p, o, []float64{0, 1e308, 0, 0}, z, 2, 2, 0.1, 0, 1}, {[]float64{1e308, -1e308}, o, []float64{0, 1, 0, 0}, z, 2, 1, 0.1, 0, 1},
		{p, []float64{1e308, 1e308}, z, z, 2, 1, 2, 0, 1}, {p, []float64{1e308, 1e308}, z, z, 2, 1, 2, 1, 1},
	}
	for i, c := range cases {
		if !math.IsNaN(steadyStateR(c.p, c.o, c.k, c.a, c.n, c.scale, c.dt, c.transient, c.measure)) {
			t.Fatalf("invalid case %d accepted", i)
		}
	}
}

func TestExportedLengthMetadataAndLegacyABI(t *testing.T) {
	p, o, k, a := []float64{0, math.Pi / 2}, make([]float64, 2), []float64{0, 5e-31, 5e-31, 0}, make([]float64, 4)
	pp := (*_Ctype_double)(unsafe.Pointer(&p[0]))
	op := (*_Ctype_double)(unsafe.Pointer(&o[0]))
	kp := (*_Ctype_double)(unsafe.Pointer(&k[0]))
	ap := (*_Ctype_double)(unsafe.Pointer(&a[0]))
	expected := math.Cos((math.Pi/2 - 1) / 2)
	if value := float64(SteadyStateRV2(pp, op, kp, ap, 2, 2, 4, 4, 2, 1, 1e30, 0, 1)); math.Abs(value-expected) > 2e-15 {
		t.Fatal("V2 calculation")
	}
	if value := float64(SteadyStateR(pp, op, kp, ap, 2, 1, 1e30, 0, 1)); math.Abs(value-expected) > 2e-15 {
		t.Fatal("legacy calculation")
	}
	for _, lengths := range [][4]uint64{{1, 2, 4, 4}, {2, 1, 4, 4}, {2, 2, 3, 4}, {2, 2, 4, 3}} {
		if !math.IsNaN(float64(SteadyStateRV2(pp, op, kp, ap, _Ctype_ulonglong(lengths[0]), _Ctype_ulonglong(lengths[1]), _Ctype_ulonglong(lengths[2]), _Ctype_ulonglong(lengths[3]), 2, 1, 0.1, 0, 1))) {
			t.Fatal("invalid extents accepted")
		}
	}
	if !math.IsNaN(float64(SteadyStateRV2(nil, op, kp, ap, 2, 2, 4, 4, 2, 1, 0.1, 0, 1))) {
		t.Fatal("nil pointer accepted")
	}
	if !math.IsNaN(float64(SteadyStateRV2(nil, nil, nil, nil, 0, 0, 0, 0, -1, 1, 0.1, 0, 1))) {
		t.Fatal("negative n accepted")
	}
	if !math.IsNaN(float64(SteadyStateRV2(nil, nil, nil, nil, 0, 0, 0, 0, 1<<62, 1, 0.1, 0, 1))) {
		t.Fatal("overflowing matrix admitted")
	}
	if !math.IsNaN(float64(SteadyStateR(nil, nil, nil, nil, 2, 1, 0.1, 0, 1))) {
		t.Fatal("legacy nil admitted")
	}
}
