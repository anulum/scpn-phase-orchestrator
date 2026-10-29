// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Embedding mutual-information binning tests

package main

import (
	"math"
	"testing"
)

// Four samples over [0.1, 0.30000000000000004] with four bins have edges
// 0.1, 0.15000000000000002, 0.2, 0.25 and 0.30000000000000004. The samples
// 0.2 and 0.15000000000000002 lie exactly on interior edges, so each bin holds
// one sample. At lag 0 the joint histogram is diagonal and the mutual
// information equals the entropy log(4). Dividing by the bin width instead
// places 0.2 in the second bin and yields 1.5*log(2).
func TestMutualInformationCountsValuesOnInteriorEdges(t *testing.T) {
	signal := []float64{0.30000000000000004, 0.1, 0.2, 0.15000000000000002}
	if got := mutualInformation(signal, 0, 4); got != math.Log(4) {
		t.Fatalf("mutual information = %v, want log(4) = %v", got, math.Log(4))
	}
}

// The last edge is inclusive and an interior value equal to an edge opens the
// next bin, as in the NumPy histogram used by the Python reference.
func TestHistogramBinUsesRightOpenInteriorAndClosedLastBin(t *testing.T) {
	edges := histogramEdges(0.0, 4.0, 4)
	cases := []struct {
		value float64
		bin   int
	}{{0.0, 0}, {0.999, 0}, {1.0, 1}, {3.0, 3}, {4.0, 3}}
	for _, c := range cases {
		if got := histogramBin(edges, c.value); got != c.bin {
			t.Fatalf("bin(%v) = %d, want %d", c.value, got, c.bin)
		}
	}
}

// Disjoint window ranges are binned separately for x and y.
func TestMutualInformationUsesSeparateWindowRanges(t *testing.T) {
	signal := []float64{0, 1, 0, 1, 10, 20}
	// lag 4: x = [0, 1], y = [10, 20]; with two bins each pair is in its own
	// cell, so the joint and marginal probabilities give log(2).
	if got := mutualInformation(signal, 4, 2); math.Abs(got-math.Log(2)) > 1e-15 {
		t.Fatalf("mutual information = %v, want log(2)", got)
	}
}
