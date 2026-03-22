<!--
SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — FEP-Kuramoto Correspondence
-->

# Mathematical Correspondence: SCPN Phase Dynamics and the Free Energy Principle

**Status:** This is a mathematical correspondence, not a philosophical claim.

## Scope

The SCPN Phase Orchestrator implements Kuramoto-type coupled oscillator dynamics. The Free Energy Principle (FEP) describes systems that minimize variational free energy (Friston, 2010). Both frameworks share structural equations that admit a precise mapping at the level of their state-update rules. This document states that mapping, distinguishes what is formally proven from what is analogical, and identifies the limits of the correspondence.

## Formal Mapping

| SCPN Quantity | Symbol | FEP Quantity | Symbol | Status |
|---|---|---|---|---|
| Phase state of oscillator n | θ_n | Sufficient statistic (mean of recognition density) | μ_n | **Exact** — both are the dynamical variables updated by gradient flow |
| Coupling matrix | K_ij | Precision matrix (inverse covariance of generative model) | Π_ij | **Exact under Laplace approx.** — K_ij weights the influence of oscillator j on i; Π_ij weights prediction error in the same role (Friston 2010, Eq. 4) |
| Phase difference sin(θ_j − θ_i) | sin(Δθ) | Prediction error | ε_ij | **Structural** — both drive the gradient flow; the sinusoidal nonlinearity in Kuramoto corresponds to the likelihood gradient under a von Mises generative model |
| Global order parameter R | R | Negative free energy −F | −F | **Analogical** — R measures coherence (low prediction error across the population); −F measures model accuracy minus complexity. Both increase as the system self-organises. The functional forms differ. |
| Natural frequency ω_i | ω_i | Prior expectation (drift term in generative model) | f(μ) | **Exact** — both appear as bias terms in the dynamics, representing the system's "expectation" of its own evolution absent coupling |

## Derivation Sketch

The standard Kuramoto model:

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)

Under the FEP with a Laplace (Gaussian) approximation to the posterior and a von Mises likelihood p(y_j | θ_i) ∝ exp(κ cos(y_j − θ_i)), the gradient descent on free energy yields:

    dμ_i/dt = f(μ_i) + Σ_j Π_ij ∂/∂μ_i [cos(y_j − μ_i)]
            = f(μ_i) + Σ_j Π_ij sin(y_j − μ_i)

Setting μ_i = θ_i, f(μ_i) = ω_i, Π_ij = K_ij, and y_j = θ_j recovers the Kuramoto equation exactly. This is a theorem about the two equation systems, not a claim about consciousness or phenomenology.

## What Is Formally Proven

1. **Equation-level equivalence.** The Kuramoto update rule is a special case of FEP gradient descent under von Mises likelihood and Laplace approximation. This follows directly from computing ∂F/∂μ.
2. **Precision-coupling identity.** The coupling matrix K plays the same algebraic role as the precision matrix Π. Increasing K_ij is formally equivalent to increasing the precision (confidence) that oscillator i assigns to the signal from oscillator j.
3. **Prediction error as phase difference.** The sin(θ_j − θ_i) term is the gradient of the negative log-likelihood under von Mises, which is the definition of prediction error in the FEP.

## What Is Analogical (Not Proven)

1. **R ↔ −F.** The global order parameter R increases with synchronisation, and −F increases with model fit. These trend together but are not equal: R is a single complex-valued mean; F is an integral over the full posterior. The monotonic relationship holds empirically in simulations but lacks a closed-form proof for heterogeneous frequencies.
2. **Biological interpretation.** The mapping is purely mathematical. It does not imply that physical oscillator networks "perceive" or "infer" in any cognitive sense.
3. **Higher-order terms.** The correspondence holds for pairwise (first-order Kuramoto) coupling. Extensions to simplicial coupling (Tanaka 2011, Millan 2020) require higher-order FEP formulations that are not yet fully established.

## Implementation

`VariationalPredictor` in `src/scpn_phase_orchestrator/upde/prediction.py` implements this correspondence numerically:

- `free_energy()` computes F = Σ ε² Π / 2 + Σ log Π
- `update()` performs gradient descent on F to update sufficient statistics μ
- `precision_weighted_coupling()` returns the precision matrix in the role of K_ij

Tests in `tests/test_prediction.py` verify:
- Free energy decreases with learning (gradient descent convergence)
- Precision increases with consistent input (low-variance ⇒ high confidence)
- Precision ∝ coupling strength (the core FEP-Kuramoto identity)

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.
- Breakspear, M., Heitmann, S., & Daffertshofer, A. (2010). Generative models of cortical oscillations. *Frontiers in Human Neuroscience*, 4, 190.
- Tanaka, H.-A., Lichtenberg, A. J., & Oishi, S. (1997). First order phase transition resulting from finite inertia in coupled oscillator systems. *Physical Review Letters*, 78(11), 2104.
