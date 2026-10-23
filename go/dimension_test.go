// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Dimension native regression tests

package main

import (
	"math"
	"testing"
)

func TestIntegralExtremeFiniteAndStrictRadius(t *testing.T) {
	for _, tc := range []struct{ point, radius, expected float64 }{
		{math.MaxFloat64 / 2, math.MaxFloat64, 1},
		{1e-200, 5e-201, 0},
		{1, 1, 0},
		{1, math.Nextafter(1, math.Inf(1)), 1},
		{0, 0, 0},
		{0, math.SmallestNonzeroFloat64, 1},
	} {
		out := []float64{-1}
		correlationIntegral([]float64{0, tc.point}, 2, 1, []int64{0}, []int64{1}, []float64{tc.radius}, out)
		if out[0] != tc.expected {
			t.Fatalf("point=%g radius=%g: got %g, expected %g", tc.point, tc.radius, out[0], tc.expected)
		}
	}
}

func TestKaplanYorkeCompensatedPrefixes(t *testing.T) {
	spectrum := []float64{math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64}
	if got := kaplanYorkeDimension(spectrum); got != 4 {
		t.Fatalf("finite cancellation dimension: got %g", got)
	}
	for _, scale := range []float64{1, 1e-305, 1e305} {
		if got := kaplanYorkeDimension([]float64{-2 * scale, 0, scale}); got != 2.5 {
			t.Fatalf("scale=%g: got %g", scale, got)
		}
	}
	if !math.IsNaN(kaplanYorkeDimension([]float64{math.Inf(1)})) {
		t.Fatal("nonfinite native spectrum must not produce a dimension")
	}
}
