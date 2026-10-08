# JAX local-coherence contract

`scpn_phase_orchestrator.nn.chimera` provides instantaneous JAX diagnostics on
nonempty finite `(N,)` phases and `(N,N)` coupling. It retains its own model:

| Function | Result |
|---|---|
| `local_order_parameter(phases, K)` | `(N,)` local-order magnitude |
| `chimera_index(phases, K)` | Scalar population variance of local-order values |
| `detect_chimera(phases, K, coherent_threshold=0.8, incoherent_threshold=0.3)` | Two `(N,)` boolean masks |

Neighbours satisfy `K != 0`: negative and self edges count, positive amplitudes
are unweighted. Empty neighbourhoods return zero. The support mask uses float32;
phasor precision follows the phase input and JAX's precision configuration.
The exact magnitude factorization removes the centre phasor before summation,
preventing overflow of opposite finite phase differences.

For a nonempty finite population, variance lies in `[0,0.25]`. Zero variance
means equal local-order values, not necessarily global synchrony. A snapshot's
variance does not establish a persistent dynamical chimera. An empty population's
variance follows JAX's empty reduction and is not a finite diagnostic.

Coherent masks use `R >= coherent_threshold`; incoherent masks use
`R <= incoherent_threshold`. They are disjoint only for ordered thresholds
`coherent_threshold > incoherent_threshold`; this JAX surface does not perform
host-side threshold or measurement validation.

JIT and phase gradients are supported on fixed adjacency away from a vanishing
neighbourhood phasor, whose magnitude is nondifferentiable. `K != 0` is a hard
support decision: coupling-amplitude gradients are zero on fixed topology,
and gradients do not define a search across topology changes. Threshold masks
are discrete. Smooth downstream losses or other models require their own
contracts.

[The CPU monitor](api/monitor_chimera.md) instead selects positive non-self
edges, reports a boundary fraction, and uses strict 0.7/0.3 thresholds. Equal
local-order outputs on a positive zero-diagonal graph do not imply equality of
their index or classification models.

`tests/test_nn_chimera.py` exercises actual JAX calls, JIT, finite extreme angles,
phase gradients, zero coupling-amplitude gradients, signed/self-edge semantics,
ordered masks and exact threshold behavior. The current comparison labels JAX
separately and synchronizes JIT results before stopping each timer.
