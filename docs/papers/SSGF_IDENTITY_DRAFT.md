# The Sextuple Identity: SSGF as Free Energy, Social Welfare, Autopoiesis, Strange Loop, Stigmergy, and Ethical Optimization

**Status: Theoretical framework paper. Not experimentally validated.**

Miroslav Šotek (ORCID: 0009-0009-3560-0851)
Anulum Research

---

## Abstract

The Self-Structuring Geometry Field (SSGF) is a computational framework
in which a latent vector z parameterizes a coupling matrix W(z) for a
network of Kuramoto oscillators. The cost functional
U_total = w_1(1−R) − w_2 λ_2(L) + w_3‖W‖_1/N² + w_4‖W−W^T‖_F/N is
minimized via gradient descent on z, producing coupling topologies adapted
to the current phase state. We identify six formal correspondences between
SSGF and established theoretical frameworks: (1) free energy minimization
(Friston), (2) Harsanyi social welfare aggregation, (3) autopoietic
self-production (Varela), (4) strange loop self-reference (Hofstadter),
(5) stigmergic indirect coordination (Grassé), and (6) cybernetic ethical
optimization (Wiener). Each correspondence is stated as a structural
mapping between SSGF quantities and quantities in the target framework,
with precise conditions under which the mapping holds and where it breaks
down. These are mathematical analogies, not empirical equivalences.

## 1. Introduction

SSGF was developed as the geometry optimization layer of the SCPN Phase
Orchestrator, a Kuramoto-based control compiler. The outer loop
z → W(z) → microcycle → cost → gradient → z produces coupling topologies
without external supervision. During development, structural parallels
to six disparate theoretical traditions became apparent. This paper makes
those parallels explicit, states each as a formal mapping, and identifies
the limits of each analogy.

The goal is not to claim that SSGF "is" free energy minimization or
autopoiesis. The goal is to show that a single cost-minimization
architecture on oscillator coupling geometry admits interpretations across
multiple theoretical vocabularies, and to explore what this convergence
does and does not imply.

## 2. SSGF Architecture

The SSGF cycle:

1. A latent vector z ∈ R^d is decoded into a coupling matrix W(z) ∈ R^{N×N}
   via a spectral decoder that ensures W has controlled eigenvalue structure.
2. The Kuramoto network runs for M microcycles under W, producing a phase
   trajectory {θ(t)}.
3. Four cost terms are computed:
   - C_1 = 1 − R (synchronization deficit)
   - C_2 = −λ_2(L(W)) (negative algebraic connectivity)
   - C_3 = ‖W‖_1 / N² (sparsity penalty)
   - C_4 = ‖W − W^T‖_F / N (asymmetry penalty)
4. U_total = Σ_k w_k C_k is minimized by ∇_z U_total, updating z.
5. The updated z decodes to a new W, and the cycle repeats.

The system produces its own coupling topology from the consequences of
its own dynamics. No external topology is prescribed.

## 3. The Six Identities

### 3.1 Identity I: Free Energy Minimization

**Source framework.** The Free Energy Principle (FEP; Friston, 2010;
Friston et al., 2006) states that self-organizing systems minimize
variational free energy F = E_q[ln q(s) − ln p(o, s)], where q is a
recognition density over hidden states s and p(o, s) is a generative
model. Minimizing F drives q toward the posterior p(s|o) and drives
the system's actions toward observations consistent with its model.

**Mapping.** In SSGF:
- Hidden states s ↔ coupling geometry W(z)
- Observations o ↔ phase dynamics {θ(t)}
- Free energy F ↔ U_total
- Gradient descent ∂z/∂t = −∇_z U_total ↔ free energy gradient flow
- The "generative model" is implicit: the spectral decoder p(W|z) and
  the Kuramoto dynamics p(θ|W) compose to p(θ|z)

U_total plays the role of F: it quantifies the discrepancy between the
current coupling topology and the topology that would produce desired
phase behavior (high R, high connectivity, sparse, symmetric). Gradient
descent on z minimizes this discrepancy.

**Limits of the analogy.** FEP free energy is defined over probability
distributions; U_total is a deterministic cost on point estimates. SSGF
does not maintain an explicit recognition density q(z) — it optimizes a
single z. The stochastic extension (Langevin noise on z; Gardiner, 2009)
partially bridges this gap by implicitly sampling from
p(z) ∝ exp(−U_total/T), but this is Boltzmann sampling, not variational
inference. The analogy holds at the gradient-flow level; it does not
extend to the full Bayesian structure of FEP.

### 3.2 Identity II: Harsanyi Social Welfare Aggregation

**Source framework.** Harsanyi's utilitarian theorem (Harsanyi, 1955)
shows that if social preferences satisfy the expected utility axioms and
the Pareto condition, then the social welfare function is a weighted sum
of individual utilities: W = Σ_i w_i U_i.

**Mapping.** In SSGF:
- Individual utilities U_i ↔ individual cost terms C_k (synchronization,
  connectivity, sparsity, symmetry)
- Welfare weights w_i ↔ cost weights (w_1, w_2, w_3, w_4)
- Social welfare W ↔ U_total = Σ_k w_k C_k
- Social planner ↔ gradient descent on z

Each cost term represents a different "stakeholder" objective:
C_1 advocates for synchronization, C_2 for connectivity, C_3 for
parsimony, C_4 for reciprocity. The weights encode the relative
importance of each objective. Minimizing U_total is Harsanyi aggregation
of these competing objectives.

**Limits of the analogy.** Harsanyi aggregation assumes cardinal,
interpersonally comparable utilities. The SSGF cost terms have no
natural common scale; the weights are chosen by the designer. There is
no Pareto condition to derive the weights endogenously — they are
external parameters. The aggregation is formally identical in structure
but lacks the axiomatic derivation that gives Harsanyi's result its
normative force.

### 3.3 Identity III: Autopoiesis

**Source framework.** Autopoiesis (Maturana & Varela, 1980; Varela,
1979) defines a living system as a network of processes that produces the
components which constitute the network itself. The key property:
operational closure — the system's products are its own processes.

**Mapping.** In SSGF:
- The network of processes ↔ the SSGF cycle
  (z → W → microcycle → cost → gradient → z)
- Components ↔ the coupling matrix W
- Operational closure ↔ the geometry produces dynamics that produce costs
  that produce gradients that produce geometry

The SSGF cycle is self-producing: the coupling topology W is both the
output of the optimization (decoded from z) and the structural
precondition for the dynamics that drive the optimization. W produces
the phase dynamics that produce the cost signal that modifies W.

**Limits of the analogy.** Autopoiesis requires that the system produce
its own boundary. SSGF does not produce its own boundary — the number
of oscillators N, the z-dimension d, and the cost function structure are
externally specified. The self-production is limited to the coupling
topology, not the system's organizational boundary. Varela distinguished
autopoiesis from mere self-organization; SSGF is closer to the latter.

### 3.4 Identity IV: Strange Loop

**Source framework.** Hofstadter (1979, 2007) defines a strange loop as
a hierarchical system in which moving through levels eventually returns
to the starting level. The paradigmatic example: Gödel's theorem, where
a formal system encodes statements about itself.

**Mapping.** In SSGF:
- Hierarchy levels: geometry (W) → dynamics (θ) → cost (U) → gradient
  (∇_z U) → geometry (W)
- The "downward" direction: geometry constrains dynamics
- The "upward" direction: dynamics produce costs that modify geometry
- The loop: geometry → dynamics → cost → geometry
- Self-reference: the coupling structure is both the object being
  evaluated (through C_1-C_4) and the entity being modified (through
  gradient on z)

The CyberneticClosure module in SPO explicitly implements this loop,
with the comment: "The loop is a strange loop (Hofstadter): geometry →
dynamics → cost → gradient → geometry."

**Limits of the analogy.** Hofstadter's strange loops involve
self-reference in a logical or representational sense — a system
representing itself within its own formalism. SSGF's loop is a
dynamical feedback cycle, not a representational self-model. The system
does not "know" it is modifying its own coupling; it follows a gradient.
The loop is strange in the dynamical sense but lacks the self-awareness
that Hofstadter considers central to consciousness-generating strange
loops.

### 3.5 Identity V: Stigmergy

**Source framework.** Stigmergy (Grassé, 1959; Theraulaz & Bonabeau,
1999) is indirect coordination through environmental modification.
Termites deposit pheromones; the pheromone field guides subsequent
construction. The agents do not communicate directly — they communicate
through the shared medium they modify.

**Mapping.** In SSGF:
- Agents ↔ individual oscillators
- Shared medium ↔ coupling matrix W
- Pheromone deposition ↔ phase dynamics under W produce cost gradients
  that modify W
- Stigmergic trace ↔ the current W encodes the history of optimization

Each oscillator does not individually modify W; the modification happens
through the aggregate effect of all oscillators' dynamics on the cost
function. The coupling matrix serves as a shared medium that mediates
coordination: oscillator i affects oscillator j not by direct
communication but by contributing to the cost gradient that reshapes
the W through which both interact.

**Limits of the analogy.** Classical stigmergy involves spatially
localized traces and decentralized agents with independent decision-making.
SSGF optimization is centralized — a single gradient step updates all
of z simultaneously. The oscillators have no agency; they follow the
Kuramoto ODE. The stigmergic structure exists at the level of the cost
landscape, not at the level of individual agent decisions.

### 3.6 Identity VI: Cybernetic Ethical Optimization

**Source framework.** Wiener (1950, 1954) argued that control systems
require ethical constraints to prevent pathological optimization. Modern
formulations use Control Barrier Functions (CBFs; Ames et al., 2017) to
enforce safety constraints as hard boundaries in the optimization
landscape.

**Mapping.** In SSGF:
- Ethical Lagrangian ↔ L_ethical = U_total + w_c15 · C15_sec
- SEC functional ↔ J_sec = α·R + β·K_norm + γ·Q − ν·S_dev
- CBF constraint penalties ↔ Φ_ethics = Σ max(0, g_k)²
- Wiener's cybernetic ethics ↔ the C15_sec term penalizes coupling
  configurations that violate safety, connectivity, or fairness
  constraints

The ethical cost module (`ssgf/ethical.py`) adds a fifteenth cost term
that encodes normative constraints: minimum synchronization (R ≥ R_min),
minimum connectivity (λ_2 ≥ λ_min), and bounded coupling strength
(K ≤ K_max). These constraints are Wiener-type ethical boundaries:
the system must not synchronize at the cost of structural stability,
and must not achieve connectivity through unbounded coupling.

**Limits of the analogy.** "Ethics" in this context means constraint
satisfaction within a predetermined normative framework. The system does
not reason about ethical principles; it minimizes a penalty function.
The normative content is entirely in the designer's choice of constraints
and weights. Calling this "ethical optimization" is defensible in
Wiener's technical sense but should not be confused with moral reasoning.

## 4. Mathematical Correspondence

The six identities share a common mathematical structure:

| Framework | System state | Objective | Update rule |
|-----------|-------------|-----------|-------------|
| SSGF | z ∈ R^d | U_total(z) | dz/dt = −∇_z U |
| Free energy | q(s) | F[q] | δq/δt = −δF/δq |
| Harsanyi | social allocation | W = Σ w_i U_i | planner maximizes W |
| Autopoiesis | organization | viability | operational closure |
| Strange loop | level in hierarchy | — | tangled hierarchy traversal |
| Stigmergy | medium state | — | trace deposition/reading |
| Cybernetic ethics | control policy | L_ethical | constrained optimization |

The first three (SSGF, FEP, Harsanyi) share the structure of gradient
descent on a weighted sum of terms — they are instances of multi-objective
optimization. The middle two (autopoiesis, strange loop) share the
structure of self-referential process cycles. The last (stigmergy) shares
the structure of mediated indirect interaction. Cybernetic ethics adds
constraint boundaries to the optimization.

The convergence suggests that self-structuring dynamics on oscillator
coupling geometry naturally admit multiple interpretive lenses because
the SSGF cycle instantiates a generic pattern: a system that modifies
its own interaction structure based on the consequences of that structure.
This pattern recurs across disciplines because it is a fundamental
organizational motif, not because the disciplines are studying the same
phenomenon.

## 5. Discussion

### 5.1 What the Convergence Means

The six identities are structural correspondences, not ontological
claims. SSGF does not "perform" autopoiesis any more than a thermostat
"performs" homeostasis — but the mathematical structure of feedback,
self-modification, and convergence is shared. The value of identifying
these correspondences is twofold:

1. **Conceptual transfer.** Results from one framework can suggest
   hypotheses in others. For example, the FEP interpretation suggests
   that adding Bayesian structure to SSGF (maintaining a distribution
   q(z) rather than a point estimate) would improve robustness to
   multimodal cost landscapes.

2. **Unification.** If a single computational mechanism admits six
   coherent interpretations, those interpretations may reflect aspects
   of a common underlying mathematical structure rather than independent
   phenomena.

### 5.2 What It Does Not Mean

The correspondences do not validate the source frameworks. SSGF's
interpretability as free energy minimization does not confirm FEP as a
theory of brain function. The strange loop interpretation does not
establish that SSGF is conscious. The ethical cost term does not make
the system a moral agent.

### 5.3 Open Questions

- Does the Boltzmann extension (Langevin noise on z) produce qualitatively
  different coupling topologies than deterministic gradient descent?
  If so, does the FEP interpretation predict which topologies emerge?
- Can the Harsanyi weights be derived from a Pareto condition on the
  cost terms, making the aggregation endogenous?
- Does the SSGF cycle satisfy a formal definition of operational closure
  (e.g., Rosen's (M,R)-systems)?

## References

- Ames, A.D. et al. (2017). Control barrier function based quadratic programs for safety critical systems. *IEEE Trans. Autom. Control* 62(8), 3861-3876.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Rev. Neurosci.* 11, 127-138.
- Friston, K. et al. (2006). A free energy principle for the brain. *J. Physiol. Paris* 100(1-3), 70-87.
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences.* 4th ed. Springer.
- Grassé, P.-P. (1959). La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp. *Insectes Sociaux* 6, 41-80.
- Harsanyi, J.C. (1955). Cardinal welfare, individualistic ethics, and interpersonal comparisons of utility. *J. Polit. Econ.* 63(4), 309-321.
- Hofstadter, D.R. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid.* Basic Books.
- Hofstadter, D.R. (2007). *I Am a Strange Loop.* Basic Books.
- Maturana, H.R. & Varela, F.J. (1980). *Autopoiesis and Cognition: The Realization of the Living.* Reidel.
- Theraulaz, G. & Bonabeau, E. (1999). A brief history of stigmergy. *Artificial Life* 5(2), 97-116.
- Varela, F.J. (1979). *Principles of Biological Autonomy.* Elsevier/North-Holland.
- Wiener, N. (1950). *The Human Use of Human Beings.* Houghton Mifflin.
- Wiener, N. (1954). *Cybernetics: Or Control and Communication in the Animal and the Machine.* 2nd ed. MIT Press.
