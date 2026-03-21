# A Dual Criterion for Consciousness Detection: Kuramoto Order Parameter and Persistent Homology

**Status: Theoretical proposal. Not empirically validated.**

Miroslav Šotek (ORCID: 0009-0009-3560-0851)
Anulum Research

---

## Abstract

The Kuramoto order parameter R measures global phase coherence in coupled
oscillator networks but does not distinguish between structurally trivial
synchronization (uniform lock) and topologically nontrivial integration
across network modules. We propose a dual criterion that supplements R with
p_h1, a scalar derived from H1 persistent homology of delay-embedded phase
dynamics. The claim is that consciousness — operationalized as integrated
information flow across functionally differentiated subsystems — requires
both sufficient coherence (R within a metastable band) and nontrivial
first homology (p_h1 above a threshold). A Topological Consciousness
Boundary Observable (TCBO) implementing this dual gate is described, with
the threshold p_h1 = 0.72 calibrated to operate in the R ~ 0.4-0.8
metastable regime. This is a theoretical framework; no experimental
validation is presented.

## 1. Introduction

### 1.1 The Insufficiency of R

The Kuramoto order parameter R = |N^{-1} Σ_j exp(iθ_j)| measures global
phase coherence in a population of N coupled oscillators (Kuramoto, 1975).
R = 1 indicates perfect synchrony; R ≈ 0 indicates incoherence. A large
body of neuroscience work uses R and related phase-locking indices
(Lachaux et al., 1999) to quantify neural synchrony, often interpreting
high R as a correlate of conscious processing.

The problem: R does not capture topology. Two networks can have identical
R values while differing in integration structure. A ring of oscillators
locked at identical phases (R = 1) and a system of functionally
differentiated modules transiently synchronizing across bridges (R ≈ 0.6)
produce different R values, yet the second configuration — with its modular
structure temporarily integrated — more closely matches theoretical
accounts of consciousness (Tononi, 2004; Dehaene & Changeux, 2011).

Fully synchronized states (R > 0.95) correspond to seizure-like dynamics
in neural systems, not conscious processing. Consciousness appears to
reside at intermediate synchronization levels (R ~ 0.4-0.8), where the
system is neither incoherent nor rigidly locked (Tagliazucchi et al., 2012).

### 1.2 The Topological Gap

Integrated Information Theory (IIT; Tononi et al., 2016) addresses
integration through Φ, but Φ computation is intractable for systems
beyond ~20 elements (NP-hard; Tegmark, 2016). Global Workspace Theory
(GWT; Baars, 1988; Dehaene et al., 2014) invokes "ignition" without
quantifying the topological structure of the global broadcast.

Persistent homology offers a computationally tractable middle ground.
H0 captures connected components; H1 captures loops (cycles that are not
boundaries). In the context of phase dynamics, an H1 feature with long
persistence indicates a stable cyclic information pathway — a topological
structure absent from trivially synchronized or fully incoherent states.

### 1.3 The Dual Criterion

We propose:

**A system exhibits dynamical signatures consistent with consciousness
when R ∈ [R_lo, R_hi] (metastable coherence) AND p_h1 > τ_h1
(nontrivial first homology).**

The parameters R_lo = 0.4, R_hi = 0.8, and τ_h1 = 0.72 are derived from
SCPN framework calibration. These are proposed thresholds, not empirically
determined constants.

## 2. Methods

### 2.1 Phase Dynamics Substrate

The oscillator network follows the extended Kuramoto equation:

dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i + α_ij) + η_i(t)

where ω_i are natural frequencies drawn from a Lorentzian distribution
with half-width Δ and center ω_0, K_ij is the coupling matrix, α_ij are
Sakaguchi-Kuramoto phase frustration terms, and η_i(t) is Gaussian white
noise with intensity σ².

### 2.2 Delay Embedding

Given N phase channels sampled at times t_1, ..., t_T, each channel θ_i
is delay-embedded into R^d using embedding dimension d and delay τ_emb:

x_i(t) = [θ_i(t), θ_i(t − τ_emb), ..., θ_i(t − (d−1)τ_emb)]

The concatenation X(t) ∈ R^{Nd} is a point in the full delay-embedded
state space. A sliding window of length W produces a point cloud
{X(t_k)}_{k=1}^{W} on which persistence is computed.

### 2.3 H1 Persistent Homology

The Vietoris-Rips filtration VR_ε(X) at scale ε includes a simplex
[x_{i_0}, ..., x_{i_k}] whenever all pairwise distances are ≤ ε. As ε
increases from 0 to ∞, topological features appear (birth) and disappear
(death). The H1 persistence diagram records (birth, death) pairs for
1-cycles.

Let L_max = max_j (death_j − birth_j) be the maximum H1 lifetime. The
scalar p_h1 is defined as the logistic squashing:

p_h1 = σ(β · (L_max − L_0))

where β = 8.0 is a steepness parameter and L_0 is a centering constant.
This maps L_max to [0, 1].

Implementation uses the ripser library (Tralie et al., 2018) for
Vietoris-Rips computation. A fallback based on pairwise Phase-Locking
Values (PLV) approximates p_h1 when ripser is unavailable, though this
approximation lacks the full topological content.

### 2.4 TCBO Gate

The Topological Consciousness Boundary Observable (TCBO) evaluates:

is_conscious = (p_h1 > τ_h1)

with τ_h1 = 0.72 by default. This operates within the R ~ 0.4-0.8
metastable band. At R > 0.95 (full synchrony), H1 features tend to
collapse — the point cloud degenerates to a neighborhood of a single
point, producing no persistent 1-cycles. At R < 0.2 (incoherence),
the point cloud fills space uniformly, producing many short-lived H1
features but no dominant persistent cycle. The metastable regime produces
H1 features with intermediate persistence, and the τ_h1 threshold
discriminates between those with structurally significant loops and those
without.

## 3. Results

**Placeholder.** Empirical results will require:

1. Synthetic Kuramoto networks with controlled modular structure, varying
   N (50-1000), modularity Q, and inter-module coupling strength. Expected
   outcome: p_h1 tracks modular integration while R remains in the
   metastable band for a range of coupling strengths.

2. EEG/MEG data from consciousness studies (e.g., anesthesia transitions,
   sleep staging). Expected outcome: p_h1 distinguishes conscious from
   unconscious states more reliably than R alone, particularly in the
   R ~ 0.5-0.7 overlap region.

3. Comparison with Φ (IIT) on small networks (N ≤ 20) where Φ is
   computable. Expected outcome: p_h1 and Φ correlate positively, with
   p_h1 computable at much larger N.

These experiments are planned but not yet conducted.

## 4. Discussion

### 4.1 What This Framework Claims

The dual criterion proposes that topological integration (H1 persistence)
and metastable coherence (R band) are jointly necessary conditions for
dynamical signatures associated with consciousness. It does not claim
sufficiency. A system satisfying both conditions exhibits structural
properties consistent with consciousness theories but may lack other
necessary conditions not captured by phase dynamics.

### 4.2 Relationship to IIT and GWT

The p_h1 measure captures a specific aspect of integration — cyclic
information flow — that Φ also measures, but through a different
mathematical lens. H1 persistence is computable in O(N^3) via the
ripser algorithm, while Φ requires exhaustive bipartition search
(exponential in N). The dual criterion trades the completeness of Φ for
computational tractability.

Global Workspace Theory's "ignition" could be operationalized as a
transition where p_h1 crosses τ_h1 from below while R enters the
metastable band — the topological structure needed for global broadcast
emerges. This connection is speculative.

### 4.3 Limitations

- The thresholds (R_lo, R_hi, τ_h1) are framework-derived, not
  empirically fitted. Different neural systems may require different
  values.
- Delay embedding parameters (d, τ_emb, W) affect the persistence
  diagram. Sensitivity analysis is needed.
- The PLV fallback approximation loses topological information and
  should not be used for scientific claims.
- Phase extraction from raw signals (Hilbert transform, wavelet)
  introduces its own biases (Pikovsky et al., 2001).
- The framework says nothing about subjective experience. It
  characterizes dynamical structure, not phenomenology.

### 4.4 The Metastability Argument

The operating point τ_h1 = 0.72 at R ~ 0.4-0.8 is motivated by the
metastability hypothesis: systems at the edge of synchronization
transition exhibit maximum dynamical repertoire (Kelso, 1995; Deco et
al., 2017). At this operating point, the system can flexibly reconfigure
its synchronization patterns — forming and dissolving functional modules —
while maintaining sufficient coherence for information integration. The
H1 persistence captures whether these reconfigurations produce
topologically nontrivial structures (loops) rather than merely transient
fluctuations.

## References

- Acebrón, J.A. et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77(1), 137-185.
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Deco, G. et al. (2017). The dynamics of resting fluctuations in the brain. *Nature Rev. Neurosci.* 18, 349-364.
- Dehaene, S. & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron* 70(2), 200-227.
- Dehaene, S. et al. (2014). Toward a computational theory of conscious processing. *Curr. Opin. Neurobiol.* 25, 76-84.
- Dörfler, F. & Bullo, F. (2014). Synchronization in complex networks of phase oscillators: A survey. *Automatica* 50(6), 1539-1564.
- Kelso, J.A.S. (1995). *Dynamic Patterns: The Self-Organization of Brain and Behavior.* MIT Press.
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420-422.
- Lachaux, J.-P. et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping* 8(4), 194-208.
- Pikovsky, A., Rosenblum, M., & Kurths, J. (2001). *Synchronization: A Universal Concept in Nonlinear Sciences.* Cambridge University Press.
- Tagliazucchi, E. et al. (2012). Criticality in large-scale brain fMRI dynamics unveiled by a novel point process analysis. *Frontiers in Physiology* 3, 15.
- Tegmark, M. (2016). Improved measures of integrated information. *PLOS Comput. Biol.* 12(11), e1005123.
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neurosci.* 5, 42.
- Tononi, G. et al. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Rev. Neurosci.* 17, 450-461.
- Tralie, C. et al. (2018). Ripser.py: A lean persistent homology library for Python. *JOSS* 3(29), 925.
