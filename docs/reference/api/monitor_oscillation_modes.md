# Inter-Area Oscillation Modes — Matrix-Pencil Damping Estimator

`monitor.oscillation_modes` recovers the electromechanical modes of a grid
ringdown — the response of a wide-area signal (a bus frequency, a tie-line angle,
the network order parameter) after a disturbance. The ringdown is a sum of damped
sinusoids; each is a *mode*, and a mode's **damping ratio** is the quantity that
matters for reliability: NERC PRC-028-1 disturbance monitoring and PRC-030-1
unexpected IBR event-mitigation evidence workflows depend on measured
disturbance records and post-event analysis, so this estimator reports the
frequency and damping evidence that reviewer packages consume.

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

## Evidence boundary

The public estimator validates the ringdown before NumPy conversion: samples
must be a one-dimensional finite real sequence with at least four values.
Boolean, complex, numeric-text, coercive object, overflowing, and broken array-
protocol inputs fail closed with `ValueError`; legitimate Python and NumPy real
numeric object scalars remain supported.

`OscillationMode` applies the same custody rule to direct construction, not only
to records emitted by the estimator. Frequency and amplitude must be finite and
non-negative, damping ratio must lie in `[-1, 1]`, phase must lie in
`[-π, π]`, and `poorly_damped` must be a canonical boolean. Accepted real scalar
types are normalised to plain JSON-safe `float` values before `to_dict()` can
publish the record. The flag itself is threshold-dependent, so a directly
constructed record can validate its type but cannot replay a caller-specific
threshold that is not stored in the record.

## Mode-family Screen

Each `OscillationMode.to_dict()` record now carries `mode_family`, computed by
`classify_oscillation_band`. The default taxonomy is an engineering review
screen, not a regulatory clause threshold:

- `aperiodic` — near-zero-frequency real-pole decay;
- `inter_area` — low-frequency area-to-area swings below 1 Hz;
- `local` — local electromechanical swings below 3 Hz;
- `sub_synchronous` — higher oscillations below the configured synchronous grid
  frequency (60 Hz by default);
- `super_synchronous` — modes at or above the configured synchronous frequency.

The cut-points are explicit function parameters so 50 Hz systems or operator
practice can override them without changing the estimator. The PRC evidence
record preserves per-mode families and aggregates `mode_family_counts`, allowing
one package to distinguish inter-area and sub-synchronous review signals while
keeping the same review-only claim boundary.

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
