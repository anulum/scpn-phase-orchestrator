# Modal Participation — Damping Controllability and Mode Shapes

`monitor.modal_participation` is the model-based companion to the data-driven
[oscillation-mode estimator](monitor_oscillation_modes.md). The matrix-pencil
estimator reads a measured ringdown and returns each mode's *damping*; this module
reads the network *model* and returns the two further things a
damping-recommendation engine needs: **where** an inter-area mode swings (mode
shape and participation factors) and **which actuator** damps it best (modal
controllability). A single ringdown signal cannot give those — they come from the
eigenstructure of the small-signal state matrix.

## The small-signal state matrix

`phase_network_jacobian` linearises the engine's Sakaguchi–Kuramoto coupling
dynamics about an operating point `θ*` (typically a synchronised fixed point
reached by running the engine):

- off-diagonal: `J_ik = K_ik cos(θ_k* − θ_i* − α_ik)`
- diagonal: `J_ii = −Σ_{k≠i} K_ik cos(θ_k* − θ_i* − α_ik) − ζ cos(Ψ − θ_i*)`

This matches the integrator's derivative exactly. For a symmetric, lag-free
network the Jacobian is symmetric negative-semidefinite — overdamped, with real
eigenvalues and no oscillation. The oscillatory inter-area modes that reliability
standards screen for appear precisely when the Sakaguchi phase lag `α` or a
directed (asymmetric) coupling break that symmetry, which is the regime the SCPN
engine runs in. A second-order swing/companion state matrix can be analysed
directly too — `analyse_network_modes` accepts any continuous-time matrix.

## Modal analysis

`analyse_network_modes` eigen-decomposes the state matrix `A` (LAPACK `geev` via
NumPy) and, for each eigenvalue `λ = σ + jω`, reports:

1. **Frequency** `f = |ω| / 2π` and **damping ratio** `ζ = −σ / |λ|` (Kundur 1994,
   §12). A growing (unstable) mode has a negative damping ratio; the marginal
   global-phase mode of a Kuramoto Jacobian appears at `f = 0`, `ζ = 0`.
2. **Mode shape** — the right eigenvector `φ_i`, unit-norm and phase-anchored, so
   `|φ_ki|` is oscillator `k`'s relative swing amplitude and `∠φ_ki` its relative
   phase. Anti-phase entries identify the machines that swing against each other.
3. **Participation factors** `p_ki = φ_ki · ψ_ik` (Pérez-Arriaga, Verghese &
   Schweppe 1982), with `ψ_i` the matching left eigenvector — dimensionless, real,
   non-negative, summing to one. `participation[k]` is how much oscillator `k`
   shapes the mode; `dominant_state` is the largest.
4. **Modal controllability** `|ψ_i · b_j|` for each column `b_j` of an optional
   input matrix `B` — how strongly input `j` can drive the mode. `dominant_input`
   is the most effective actuator: where to add damping.

Complex eigenvalues of a real matrix occur in conjugate pairs; only the
non-negative-frequency member of each pair is reported. Modes are ordered by
ascending damping ratio — the least-damped, most critical mode first. A defective
(non-diagonalisable) state matrix is rejected: its left eigenvectors, and the
participation factors built from them, do not exist.

## Relationship to `oscillation_modes`

The two pieces describe the *same* linear modes from opposite ends. For a linear
ringdown `x(t) = e^{At} x₀`, `estimate_oscillation_modes` recovers the eigenvalues
of `A` from the measured signal, and `analyse_network_modes(A)` computes them from
the model — they agree (this cross-check is part of the test suite). Use the
estimator when you only have measurements; use this module when you have the model
and need mode shape, participation, or controllability.

## NumPy floor

The analysis is one offline eigen-decomposition of a modest state matrix (LAPACK
via NumPy), not a per-step hot path, so it stays on the NumPy floor — the same
judgement as `oscillation_modes` and `autotune.freq_id`; it carries no
multi-language acceleration chain.

## Review-only

Like every monitor primitive, the analysis only reads a model and reports modes;
it never changes bindings, layers, or coupling.

::: scpn_phase_orchestrator.monitor.modal_participation
