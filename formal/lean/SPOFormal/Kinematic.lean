/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Lean kinematic safety proofs
-/

import Std

set_option autoImplicit false

namespace SPOFormal.Kinematic

/-- Fixed-point kinematic bounds for a relative-distance proof lane.

All values are natural fixed-point units chosen by the caller. For MIF-style
merge windows, one unit may be one micron or one tenth of a micron; the theorem
only requires that all runtime values use the same scale.
-/
structure KinematicBounds where
  initialTolerance : Nat
  lipschitzStepGain : Nat
  relativeVelocityStepBound : Nat
  couplingResidualStepBound : Nat
  mergeWindowTolerance : Nat
  horizonSteps : Nat
  deriving DecidableEq, Repr

/-- Per-step additive drive budget: relative velocity plus residual coupling. -/
def KinematicBounds.driveBound (cfg : KinematicBounds) : Nat :=
  cfg.relativeVelocityStepBound + cfg.couplingResidualStepBound

/-- Basic well-formedness: the initial distance budget starts inside the window. -/
def KinematicBounds.Valid (cfg : KinematicBounds) : Prop :=
  cfg.initialTolerance <= cfg.mergeWindowTolerance

instance (cfg : KinematicBounds) : Decidable cfg.Valid := by
  unfold KinematicBounds.Valid
  infer_instance

/-- Discrete Gronwall-style budget recurrence.

`gain` is a fixed-point per-step Lipschitz multiplier. `drive` is an additive
per-step disturbance bound. The recurrence is intentionally discrete and
finite-horizon so it can be mirrored by runtime monitors without importing a
continuous-time analysis stack into Lean.
-/
def gronwallBudget (initial gain drive : Nat) : Nat -> Nat
  | 0 => initial
  | k + 1 =>
      let previous := gronwallBudget initial gain drive k
      previous + gain * previous + drive

@[simp]
theorem gronwallBudget_zero {initial gain drive : Nat} :
    gronwallBudget initial gain drive 0 = initial := by
  rfl

@[simp]
theorem gronwallBudget_succ {initial gain drive k : Nat} :
    gronwallBudget initial gain drive (k + 1) =
      gronwallBudget initial gain drive k +
        gain * gronwallBudget initial gain drive k +
        drive := by
  rfl

/-- Kinematic budget associated with a configuration. -/
def KinematicBounds.budget (cfg : KinematicBounds) (k : Nat) : Nat :=
  gronwallBudget cfg.initialTolerance cfg.lipschitzStepGain cfg.driveBound k

@[simp]
theorem budget_zero {cfg : KinematicBounds} :
    cfg.budget 0 = cfg.initialTolerance := by
  rfl

@[simp]
theorem budget_succ {cfg : KinematicBounds} {k : Nat} :
    cfg.budget (k + 1) =
      cfg.budget k + cfg.lipschitzStepGain * cfg.budget k + cfg.driveBound := by
  rfl

/-- The finite-horizon Gronwall budget is monotone in the horizon. -/
theorem gronwallBudget_monotone
    {initial gain drive : Nat} :
    ∀ {k m : Nat}, k <= m ->
      gronwallBudget initial gain drive k <= gronwallBudget initial gain drive m := by
  intro k m h
  induction h with
  | refl =>
      exact Nat.le_refl _
  | step h ih =>
      exact Nat.le_trans ih (by
        rw [gronwallBudget_succ]
        exact Nat.le_trans
          (Nat.le_add_right _ _)
          (Nat.le_add_right _ _))

/-- The bundled budget is monotone in the horizon. -/
theorem KinematicBounds.budget_monotone
    {cfg : KinematicBounds} {k m : Nat}
    (h : k <= m) :
    cfg.budget k <= cfg.budget m := by
  unfold KinematicBounds.budget
  exact gronwallBudget_monotone h

/-- Runtime-facing certificate for arbitrary non-negative finite-horizon gain. -/
def KinematicBounds.budgetCertificate (cfg : KinematicBounds) : Bool :=
  cfg.budget cfg.horizonSteps <= cfg.mergeWindowTolerance

/-- Boolean certificate reflection for the general finite-horizon budget. -/
theorem budgetCertificate_eq_true_iff {cfg : KinematicBounds} :
    cfg.budgetCertificate = true ↔
      cfg.budget cfg.horizonSteps <= cfg.mergeWindowTolerance := by
  unfold KinematicBounds.budgetCertificate
  simp

/-- A true finite-horizon certificate discharges every prefix budget. -/
theorem budget_certificate_discharges_budget
    {cfg : KinematicBounds}
    (hCertificate : cfg.budgetCertificate = true) :
    ∀ k, k <= cfg.horizonSteps -> cfg.budget k <= cfg.mergeWindowTolerance := by
  intro k hk
  have hHorizon := (budgetCertificate_eq_true_iff (cfg := cfg)).mp hCertificate
  exact Nat.le_trans (KinematicBounds.budget_monotone (cfg := cfg) hk) hHorizon

/-- Zero Lipschitz gain reduces the recurrence to a linear finite-step budget. -/
theorem gronwallBudget_zero_gain_eq_linear
    {initial drive : Nat} :
    ∀ k, gronwallBudget initial 0 drive k = initial + k * drive := by
  intro k
  induction k with
  | zero => simp [gronwallBudget]
  | succ k ih =>
      calc
        gronwallBudget initial 0 drive (k + 1) =
            initial + k * drive + drive := by
          simp [gronwallBudget, ih]
        _ = initial + (k + 1) * drive := by
          simp [Nat.succ_mul, Nat.add_assoc]

/-- If a metric-distance sequence satisfies the one-step Lipschitz recurrence,
then it is bounded by the discrete Gronwall budget at every step. -/
theorem distance_bound_under_lipschitz_coupling
    (distance : Nat -> Nat)
    (cfg : KinematicBounds)
    (hInitial : distance 0 <= cfg.initialTolerance)
    (hStep : ∀ k,
      distance (k + 1) <=
        distance k + cfg.lipschitzStepGain * distance k + cfg.driveBound) :
    ∀ k, distance k <= cfg.budget k := by
  intro k
  induction k with
  | zero =>
      simpa [KinematicBounds.budget] using hInitial
  | succ k ih =>
      calc
        distance (k + 1) <=
            distance k + cfg.lipschitzStepGain * distance k + cfg.driveBound :=
          hStep k
        _ <= cfg.budget k + cfg.lipschitzStepGain * cfg.budget k +
              cfg.driveBound := by
          exact Nat.add_le_add
            (Nat.add_le_add ih (Nat.mul_le_mul_left cfg.lipschitzStepGain ih))
            (Nat.le_refl cfg.driveBound)
        _ = cfg.budget (k + 1) := by
          simp [KinematicBounds.budget]

/-- A reviewed budget table implies the merge-window spatial invariant. -/
theorem merge_window_invariant_under_bounded_coupling
    (distance : Nat -> Nat)
    (cfg : KinematicBounds)
    (hInitial : distance 0 <= cfg.initialTolerance)
    (hStep : ∀ k,
      distance (k + 1) <=
        distance k + cfg.lipschitzStepGain * distance k + cfg.driveBound)
    (hWindowBudget : ∀ k, k <= cfg.horizonSteps ->
      cfg.budget k <= cfg.mergeWindowTolerance) :
    ∀ k, k <= cfg.horizonSteps -> distance k <= cfg.mergeWindowTolerance := by
  intro k hk
  exact Nat.le_trans
    (distance_bound_under_lipschitz_coupling distance cfg hInitial hStep k)
    (hWindowBudget k hk)

/-- Zero-gain horizons can be certified from one scalar inequality. -/
theorem zero_gain_budget_within_horizon
    {cfg : KinematicBounds}
    (hGain : cfg.lipschitzStepGain = 0)
    (hWindow : cfg.initialTolerance + cfg.horizonSteps * cfg.driveBound <=
      cfg.mergeWindowTolerance) :
    ∀ k, k <= cfg.horizonSteps -> cfg.budget k <= cfg.mergeWindowTolerance := by
  intro k hk
  unfold KinematicBounds.budget
  rw [hGain]
  rw [gronwallBudget_zero_gain_eq_linear]
  exact Nat.le_trans
    (Nat.add_le_add_left (Nat.mul_le_mul_right cfg.driveBound hk) cfg.initialTolerance)
    hWindow

/-- Boolean mirror used by runtime PHA-C proof-obligation manifests. -/
def KinematicBounds.zeroGainCertificate (cfg : KinematicBounds) : Bool :=
  cfg.lipschitzStepGain == 0 &&
    cfg.initialTolerance + cfg.horizonSteps * cfg.driveBound <=
      cfg.mergeWindowTolerance

theorem zeroGainCertificate_eq_true_iff {cfg : KinematicBounds} :
    cfg.zeroGainCertificate = true ↔
      cfg.lipschitzStepGain = 0 ∧
        cfg.initialTolerance + cfg.horizonSteps * cfg.driveBound <=
          cfg.mergeWindowTolerance := by
  unfold KinematicBounds.zeroGainCertificate
  simp

/-- Runtime zero-gain certificates discharge every finite-horizon budget row. -/
theorem zero_gain_certificate_discharges_budget
    {cfg : KinematicBounds}
    (hCertificate : cfg.zeroGainCertificate = true) :
    ∀ k, k <= cfg.horizonSteps -> cfg.budget k <= cfg.mergeWindowTolerance := by
  have h := (zeroGainCertificate_eq_true_iff (cfg := cfg)).mp hCertificate
  exact zero_gain_budget_within_horizon h.1 h.2

/-- Linear finite-step invariant for the common no-Lipschitz-residual case. -/
theorem merge_window_invariant_zero_gain
    (distance : Nat -> Nat)
    (cfg : KinematicBounds)
    (hGain : cfg.lipschitzStepGain = 0)
    (hWindow : cfg.initialTolerance + cfg.horizonSteps * cfg.driveBound <=
      cfg.mergeWindowTolerance)
    (hInitial : distance 0 <= cfg.initialTolerance)
    (hStep : ∀ k, distance (k + 1) <= distance k + cfg.driveBound) :
    ∀ k, k <= cfg.horizonSteps -> distance k <= cfg.mergeWindowTolerance := by
  apply merge_window_invariant_under_bounded_coupling distance cfg hInitial
  · intro k
    rw [hGain]
    simpa using hStep k
  · exact zero_gain_budget_within_horizon hGain hWindow

/-- Boolean phase/spatial merge-window decision used by finite proof mirrors. -/
def mergeWindowLocked
    (phaseDispersion phaseTolerance spatialDistance spatialTolerance : Nat) : Bool :=
  phaseDispersion <= phaseTolerance && spatialDistance <= spatialTolerance

/-- The Boolean merge-window mirror is true exactly when both bounds hold. -/
theorem mergeWindowLocked_eq_true_iff
    {phaseDispersion phaseTolerance spatialDistance spatialTolerance : Nat} :
    mergeWindowLocked
      phaseDispersion
      phaseTolerance
      spatialDistance
      spatialTolerance = true ↔
      phaseDispersion <= phaseTolerance ∧ spatialDistance <= spatialTolerance := by
  unfold mergeWindowLocked
  simp

/-- Spatial invariant plus phase-lock evidence implies the finite merge-window
Boolean stays locked over the reviewed horizon. -/
theorem merge_window_locked_over_horizon
    (spatialDistance : Nat -> Nat)
    (phaseDispersion : Nat -> Nat)
    (cfg : KinematicBounds)
    (phaseTolerance : Nat)
    (hSpatial : ∀ k, k <= cfg.horizonSteps ->
      spatialDistance k <= cfg.mergeWindowTolerance)
    (hPhase : ∀ k, k <= cfg.horizonSteps ->
      phaseDispersion k <= phaseTolerance) :
    ∀ k, k <= cfg.horizonSteps ->
      mergeWindowLocked
        (phaseDispersion k)
        phaseTolerance
        (spatialDistance k)
        cfg.mergeWindowTolerance = true := by
  intro k hk
  exact (mergeWindowLocked_eq_true_iff).mpr ⟨hPhase k hk, hSpatial k hk⟩

end SPOFormal.Kinematic
