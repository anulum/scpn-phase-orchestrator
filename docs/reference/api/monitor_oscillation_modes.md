# Inter-Area Oscillation Modes — Matrix-Pencil Damping Estimator

`monitor.oscillation_modes` recovers the electromechanical modes of a grid
ringdown — the response of a wide-area signal (a bus frequency, a tie-line angle,
the network order parameter) after a disturbance. The ringdown is a sum of damped
sinusoids; each is a *mode*, and a mode's **damping ratio** is the quantity that
matters for reliability: North American standards (NERC PRC-028 and the proposed
PRC-030 oscillation-monitoring rules) flag a mode whose damping ratio sits below a
few percent as a poorly-damped inter-area oscillation.

## Method

`estimate_oscillation_modes` uses the **matrix-pencil method** of Hua & Sarkar
(1990):

1. Build a Hankel matrix from the ringdown samples.
2. Take its singular value decomposition (LAPACK via NumPy) and keep the dominant
   signal subspace — the model order, chosen from the singular spectrum or set
   explicitly.
3. Recover each discrete pole `z = exp((−α + j·2πf)·Δt)` as a generalised
   eigenvalue of the pencil formed from the right singular vectors.
4. Map each pole to a physical mode:
   `frequency = |∠z|·fs / 2π`, `ζ = −ln|z| / hypot(ln|z|, ∠z)`,
   and recover amplitude and phase from a least-squares Vandermonde fit.

Matrix pencil is a one-shot SVD problem rather than a polynomial root-find, so it
is far less noise-sensitive than Prony. Complex poles of a real signal occur in
conjugate pairs, which are merged into one positive-frequency mode; a real pole is
reported as a pure decay at `frequency = 0`. Modes are returned ordered by
descending amplitude, and any mode whose damping ratio is below the screening
threshold (`DEFAULT_DAMPING_THRESHOLD = 0.03`) is flagged `poorly_damped`; a
growing (unstable) mode has a negative damping ratio.

## Relationship to `autotune.freq_id`

`autotune.freq_id` runs multichannel DMD to assign oscillator channels to modal
frequencies for tuning. This estimator answers a different question for
oscillation **safety monitoring**: the modal *damping* of a single ringdown
signal. Both are one-shot offline SVD/eigen-solves on a short window (the NumPy
floor — not a per-step hot path), so neither carries the multi-language
acceleration chain.

## Review-only

Like every monitor primitive, the estimator only reads a signal and reports
modes; it never changes bindings, layers, or coupling.

::: scpn_phase_orchestrator.monitor.oscillation_modes
