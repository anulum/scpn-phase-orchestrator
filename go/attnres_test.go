// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Native phase attention contracts

package main

import (
	"math"
	"reflect"
	"testing"
	"unsafe"
)

func identityAttnresWeights(width, heads int) ([]float64, []float64) {
	projection, output := make([]float64, width*width), make([]float64, width*width)
	headWidth := width / heads
	for head := 0; head < heads; head++ {
		for feature := 0; feature < headWidth; feature++ {
			projection[head*width*headWidth+(head*headWidth+feature)*headWidth+feature] = 1
		}
	}
	for feature := 0; feature < width; feature++ {
		output[feature*width+feature] = 1
	}
	return projection, output
}

func TestAttnresOrthogonalSignedPair(t *testing.T) {
	projection, output := identityAttnresWeights(2, 1)
	result, err := attnres([]float64{0, -.3, -.3, 0}, []float64{0, math.Pi / 2},
		projection, projection, projection, output, 2, 1, -1, 1, .5)
	if err != nil || math.Abs(result[1]+.375) > 1e-12 || result[1] != result[2] || result[0] != 0 {
		t.Fatalf("analytic signed pair: %v, %v", result, err)
	}
}

func TestAttnresRejectsInvalidDomainBeforeIdentity(t *testing.T) {
	projection, output := identityAttnresWeights(2, 1)
	for _, coupling := range [][]float64{{.1, .3, .3, 0}, {0, .3, .1, 0}, {0, math.NaN(), math.NaN(), 0}} {
		if _, err := attnres(coupling, []float64{0, .4}, projection, projection, projection, output,
			2, 1, -1, 1, 0); err == nil {
			t.Fatalf("accepted malformed identity topology %v", coupling)
		}
	}
	for _, strength := range []float64{math.NaN(), math.Inf(1), -.1} {
		if _, err := attnres([]float64{0, .3, .3, 0}, []float64{0, .4}, projection,
			projection, projection, output, 2, 1, -1, 1, strength); err == nil {
			t.Fatalf("accepted invalid strength %v", strength)
		}
	}
}

func TestAttnresRejectsInvalidDimensionsAndCABI(t *testing.T) {
	projection, output := identityAttnresWeights(2, 1)
	if _, err := attnres(nil, nil, projection, projection, projection, output,
		-1, 1, -1, 1, 0); err == nil {
		t.Fatal("accepted negative count")
	}
	if AttnResModulate(nil, nil, nil, nil, nil, nil, 0, 0, -1, 1, .5, nil) == 0 {
		t.Fatal("legacy ABI accepted zero heads")
	}
	if AttnResModulateV2(nil, nil, nil, nil, nil, nil, 0, 1, 3, -1, 1, .5, nil) == 0 {
		t.Fatal("V2 ABI accepted odd width")
	}
	if AttnResModulateV2(nil, nil, nil, nil, nil, nil, 0, 1, 4, -1, 1, .5, nil) == 0 {
		t.Fatal("V2 ABI accepted missing projections")
	}
}

func TestAttnresLargeCouplingAndOverflowRefusal(t *testing.T) {
	zeros := make([]float64, 4)
	result, err := attnres([]float64{0, 1e308, 1e308, 0}, []float64{0, .4},
		zeros, zeros, zeros, zeros, 2, 1, -1, 1, .1)
	if err != nil || math.Abs(result[1]/1e308-1.05) > 1e-14 {
		t.Fatalf("representable coupling: %v, %v", result, err)
	}
	projection, output := identityAttnresWeights(2, 1)
	if _, err := attnres([]float64{0, .3, .3, 0}, []float64{0, .4}, projection,
		projection, projection, output, 2, 1, -1, math.SmallestNonzeroFloat64, .5); err == nil {
		t.Fatal("accepted unrepresentable temperature scale")
	}
}

// callAttnresABI constructs the actual named cgo pointer/control types from
// exported function metadata. It invokes the real entry point, retaining all
// owned buffers for the duration of the call; no kernel or result is replaced.
func callAttnresABI(function any, buffers [][]float64, controls []int, temperature, strength float64) int {
	entry := reflect.ValueOf(function)
	values := make([]reflect.Value, 0, entry.Type().NumIn())
	for index, buffer := range buffers[:6] {
		values = append(values, reflect.NewAt(entry.Type().In(index).Elem(), unsafe.Pointer(&buffer[0])))
	}
	for _, control := range controls {
		values = append(values, reflect.ValueOf(control).Convert(entry.Type().In(len(values))))
	}
	for _, control := range []float64{temperature, strength} {
		values = append(values, reflect.ValueOf(control).Convert(entry.Type().In(len(values))))
	}
	output := buffers[6]
	values = append(values, reflect.NewAt(entry.Type().In(len(values)).Elem(), unsafe.Pointer(&output[0])))
	return int(entry.Call(values)[0].Int())
}

func TestAttnresExportedABIsComputeAndPreserveOutputOnError(t *testing.T) {
	for _, width := range []int{2, 4, 8, 12, 16} {
		projection, outputProjection := identityAttnresWeights(width, 1)
		coupling, phases := []float64{0, -.3, -.3, 0}, []float64{.1, .7}
		output := []float64{37, 37, 37, 37}
		buffers := [][]float64{coupling, phases, projection, projection, projection, outputProjection, output}
		cosineSum := 0.0
		for harmonic := 1; harmonic <= width/2; harmonic++ {
			cosineSum += math.Cos(float64(harmonic) * .6)
		}
		norm := math.Sqrt(float64(width)/2) + 1e-12
		expected := -.3 * (1 + .5*(1+cosineSum/(norm*norm))/2)
		if status := callAttnresABI(AttnResModulateV2, buffers, []int{2, 1, width, -1}, 1, .5); status != 0 || math.Abs(output[1]-expected) > 1e-12 {
			t.Fatalf("V2 width %d: status %d, output %v", width, status, output)
		}
		if width == 8 {
			if status := callAttnresABI(AttnResModulate, buffers, []int{2, 1, -1}, 1, .5); status != 0 || math.Abs(output[1]-expected) > 1e-12 {
				t.Fatalf("legacy width8: status %d, output %v", status, output)
			}
		}
		for index := range output {
			output[index] = 37
		}
		if status := callAttnresABI(AttnResModulateV2, buffers, []int{2, 1, width, -1}, 1, math.NaN()); status == 0 {
			t.Fatal("C ABI accepted NaN strength")
		}
		for _, value := range output {
			if value != 37 {
				t.Fatalf("refused call wrote output: %v", output)
			}
		}
	}
}

func TestAttnresPublicDimensionAndNumericalRefusals(t *testing.T) {
	valid, output := identityAttnresWeights(2, 1)
	graph, phases := []float64{0, .3, .3, 0}, []float64{.4, .5}
	cases := []struct {
		name                        string
		graph, phases, q, k, v, out []float64
		n, heads, band              int
		temperature, strength       float64
	}{
		{"count-overflow", nil, nil, valid, valid, valid, output, int(^uint(0) >> 1), 1, -1, 1, .5},
		{"graph-length", []float64{0}, phases, valid, valid, valid, output, 2, 1, -1, 1, .5},
		{"phase-length", graph, nil, valid, valid, valid, output, 2, 1, -1, 1, .5},
		{"heads", graph, phases, valid, valid, valid, output, 2, 0, -1, 1, .5},
		{"temperature", graph, phases, valid, valid, valid, output, 2, 1, -1, math.Inf(1), .5},
		{"band", graph, phases, valid, valid, valid, output, 2, 1, 0, 1, .5},
		{"finite-phase", graph, []float64{math.Inf(1), 0}, valid, valid, valid, output, 2, 1, -1, 1, .5},
		{"projection-length", graph, phases, valid, nil, valid, output, 2, 1, -1, 1, .5},
		{"head-division", graph, phases, valid, valid, valid, output, 2, 3, -1, 1, .5},
		{"head-square", graph, phases, valid, valid, valid, output, 2, 4, -1, 1, .5},
		{"odd-width", graph, phases, []float64{1}, []float64{1}, []float64{1}, []float64{1}, 2, 1, -1, 1, .5},
		{"output-length", graph, phases, valid, valid, valid, nil, 2, 1, -1, 1, .5},
		{"project-overflow", graph, phases, []float64{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64, math.MaxFloat64}, valid, valid, output, 2, 1, -1, 1, .5},
		{"logit-overflow", graph, phases, []float64{1e154, 1e154, 1e154, 1e154}, []float64{1e154, 1e154, 1e154, 1e154}, valid, output, 2, 1, -1, 1, .5},
		{"norm-overflow", graph, phases, valid, valid, []float64{1e155, 1e155, 1e155, 1e155}, output, 2, 1, -1, 1, .5},
		{"coupling-overflow", []float64{0, math.MaxFloat64, math.MaxFloat64, 0}, phases, valid, valid, valid, output, 2, 1, -1, 1, .5},
	}
	for _, fixture := range cases {
		t.Run(fixture.name, func(t *testing.T) {
			if _, err := attnres(fixture.graph, fixture.phases, fixture.q, fixture.k,
				fixture.v, fixture.out, fixture.n, fixture.heads, fixture.band,
				fixture.temperature, fixture.strength); err == nil {
				t.Fatalf("accepted invalid native domain: %s", fixture.name)
			}
		})
	}
}

func TestAttnresBandedSparseGraphAndIdentity(t *testing.T) {
	projection, output := identityAttnresWeights(4, 2)
	graph := []float64{0, .3, .2, 0, .3, 0, -.1, 0, .2, -.1, 0, 0, 0, 0, 0, 0}
	phases := []float64{.1, .7, 1.8, 2.4}
	result, err := attnres(graph, phases, projection, projection, projection, output, 4, 2, 1, 1, .5)
	if err != nil || result[2] != graph[2] || result[14] != 0 || result[1] <= graph[1] {
		t.Fatalf("banded sparse graph: %v, %v", result, err)
	}
	identity, err := attnres(graph, phases, projection, projection, projection, output, 4, 2, -1, 1, 0)
	if err != nil || !reflect.DeepEqual(identity, graph) {
		t.Fatalf("identity: %v, %v", identity, err)
	}
}

func TestAttnresLargeGainPreservesCosineEndpointBounds(t *testing.T) {
	projection, output := identityAttnresWeights(2, 1)
	for index := range output {
		output[index] *= 1e120
	}
	phase := 0.1972727272727273
	for _, edge := range []float64{.3, -.3} {
		for _, antipodal := range []bool{true, false} {
			second, expected := phase, edge*(1+1e16)
			if antipodal {
				second, expected = phase+math.Pi, edge
			}
			result, err := attnres([]float64{0, edge, edge, 0}, []float64{phase, second},
				projection, projection, projection, output, 2, 1, -1, 1, 1e16)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(result[1]-expected) > 1e-12+math.Abs(expected)*2e-15 ||
				math.Signbit(result[1]) != math.Signbit(edge) ||
				math.Abs(result[1]) < math.Abs(edge) || math.Abs(result[1]) > math.Abs(edge)*(1+1e16) {
				t.Fatalf("cosine endpoint with antipodal=%v: %v, expected %v", antipodal, result, expected)
			}
		}
	}
}

func TestAttnresCABIRejectsUnrepresentableFloatBufferExtents(t *testing.T) {
	projection, outputProjection := identityAttnresWeights(2, 1)
	output := []float64{37, 37, 37, 37}
	buffers := [][]float64{{0, .3, .3, 0}, {.1, .7}, projection,
		projection, projection, outputProjection, output}
	for _, controls := range [][]int{
		{0, 1, 1 << 30, -1},
		{0, 1, (1 << 31) - 2, -1},
		{(1 << 31) - 1, 1, 2, -1},
		{65536, 1 << 29, 1 << 29, -1},
	} {
		if status := callAttnresABI(AttnResModulateV2, buffers, controls, 1, .5); status == 0 {
			t.Fatalf("accepted impossible float buffer extents %v", controls)
		}
		for _, value := range output {
			if value != 37 {
				t.Fatal("impossible extent changed output before refusal")
			}
		}
	}
}

func TestAttnresEmptyGraphDoesNotEvaluateUnusedScale(t *testing.T) {
	projection, output := identityAttnresWeights(2, 1)
	result, err := attnres(nil, nil, projection, projection, projection, output,
		0, 1, -1, math.SmallestNonzeroFloat64, .5)
	if err != nil || len(result) != 0 {
		t.Fatalf("valid empty graph has no attention calculation: %v, %v", result, err)
	}
}
