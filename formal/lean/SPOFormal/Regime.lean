/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Lean regime classifier proofs
-/

import Std

set_option autoImplicit false

namespace SPOFormal.Regime

/-- Discrete supervisor regimes, mirroring the Rust `spo_types::Regime` variants. -/
inductive Regime where
  | nominal
  | degraded
  | recovery
  | critical
  deriving DecidableEq, Repr

/-- Fixed-point representation of R=0.3 on a 1_000_000-unit interval. -/
def rCritical : Nat := 300000

/-- Fixed-point representation of R=0.6 on a 1_000_000-unit interval. -/
def rDegraded : Nat := 600000

theorem rCritical_le_rDegraded : rCritical <= rDegraded := by
  decide

theorem rCritical_lt_rDegraded : rCritical < rDegraded := by
  decide

/-- Recovering regimes must pass through recovery before returning nominal. -/
def isRecovering : Regime -> Bool
  | Regime.critical => true
  | Regime.recovery => true
  | _ => false

theorem isRecovering_eq_true_iff {regime : Regime} :
    isRecovering regime = true ↔ regime = Regime.critical ∨ regime = Regime.recovery := by
  cases regime <;> simp [isRecovering]

/-- Severity rank used by the production transition guard. -/
def regimeRank : Regime -> Nat
  | Regime.nominal => 0
  | Regime.degraded => 1
  | Regime.recovery => 2
  | Regime.critical => 3

theorem nominal_has_min_rank {regime : Regime} :
    regimeRank Regime.nominal <= regimeRank regime := by
  cases regime <;> decide

theorem critical_has_max_rank {regime : Regime} :
    regimeRank regime <= regimeRank Regime.critical := by
  cases regime <;> decide

theorem critical_strictly_above_noncritical {regime : Regime}
    (hNonCritical : regime ≠ Regime.critical) :
    regimeRank regime < regimeRank Regime.critical := by
  cases regime <;> simp [regimeRank] at hNonCritical ⊢

/-- Discrete fixed-point mirror of `classify_regime_from_summary` for finite inputs. -/
def classify (current : Regime) (meanR hardViolationCount hysteresis : Nat) : Regime :=
  if hardViolationCount > 0 then Regime.critical
  else if meanR < rCritical then Regime.critical
  else if meanR < rDegraded then
    if isRecovering current then Regime.recovery else Regime.degraded
  else if current = Regime.degraded ∧ meanR < rDegraded + hysteresis then
    Regime.degraded
  else if isRecovering current ∧ meanR < rDegraded + hysteresis then
    Regime.recovery
  else if current = Regime.critical then
    Regime.recovery
  else
    Regime.nominal

/-- Minimal transition-guard mirror for cooldown and downward hold contracts. -/
structure TransitionGuardState where
  current : Regime
  stepCounter : Nat
  lastTransition : Nat
  cooldownSteps : Nat
  holdSteps : Nat
  downwardStreak : Nat

/-- Next committed regime under the same guard order as the production FSM. -/
def transitionCore (state : TransitionGuardState) (proposed : Regime) : Regime :=
  if proposed = state.current then state.current
  else if regimeRank state.current < regimeRank proposed ∧
      proposed ≠ Regime.critical ∧
      0 < state.holdSteps ∧
      state.downwardStreak + 1 < state.holdSteps then
    state.current
  else if 0 < state.lastTransition ∧
      state.stepCounter.succ - state.lastTransition < state.cooldownSteps ∧
      proposed ≠ Regime.critical then
    state.current
  else
    proposed

theorem transitionCore_same_current {state : TransitionGuardState} :
    transitionCore state state.current = state.current := by
  unfold transitionCore
  simp

theorem transitionCore_critical_bypasses_hold_and_cooldown
    {state : TransitionGuardState} :
    transitionCore state Regime.critical = Regime.critical := by
  unfold transitionCore
  cases state.current <;> simp [regimeRank]

theorem transitionCore_soft_downward_hold_blocks
    {state : TransitionGuardState} {proposed : Regime}
    (hDifferent : proposed ≠ state.current)
    (hDownward : regimeRank state.current < regimeRank proposed)
    (hNotCritical : proposed ≠ Regime.critical)
    (hHoldEnabled : 0 < state.holdSteps)
    (hHoldPending : state.downwardStreak + 1 < state.holdSteps) :
    transitionCore state proposed = state.current := by
  unfold transitionCore
  simp [
    hDifferent,
    hDownward,
    hNotCritical,
    hHoldEnabled,
    hHoldPending,
  ]

theorem transitionCore_cooldown_blocks_noncritical
    {state : TransitionGuardState} {proposed : Regime}
    (hDifferent : proposed ≠ state.current)
    (hLastTransition : 0 < state.lastTransition)
    (hCooldown : state.stepCounter.succ - state.lastTransition < state.cooldownSteps)
    (hNotCritical : proposed ≠ Regime.critical) :
    transitionCore state proposed = state.current := by
  unfold transitionCore
  simp [
    hDifferent,
    hLastTransition,
    hCooldown,
    hNotCritical,
  ]

theorem nominal_safe_summary_never_classifies_critical
    {meanR hysteresis : Nat} (hMean : rCritical <= meanR) :
    classify Regime.nominal meanR 0 hysteresis ≠ Regime.critical := by
  unfold classify
  simp [Nat.not_lt_of_ge hMean, isRecovering]
  split <;> simp

theorem safe_summary_without_hard_violation_never_classifies_critical
    {current : Regime} {meanR hysteresis : Nat}
    (hMean : rCritical <= meanR) :
    classify current meanR 0 hysteresis ≠ Regime.critical := by
  cases current
  · exact nominal_safe_summary_never_classifies_critical hMean
  · unfold classify
    simp [Nat.not_lt_of_ge hMean, isRecovering]
    by_cases hDegraded : meanR < rDegraded
    · simp [hDegraded]
    · simp [hDegraded]
      by_cases hHolding : meanR < rDegraded + hysteresis
      · simp [hHolding]
      · simp [hHolding]
  · unfold classify
    simp [Nat.not_lt_of_ge hMean, isRecovering]
    by_cases hDegraded : meanR < rDegraded
    · simp [hDegraded]
    · simp [hDegraded]
      by_cases hHolding : meanR < rDegraded + hysteresis
      · simp [hHolding]
      · simp [hHolding]
  · unfold classify
    simp [Nat.not_lt_of_ge hMean, isRecovering]

theorem critical_never_evaluates_directly_to_nominal
    {meanR hardViolationCount hysteresis : Nat} :
    classify Regime.critical meanR hardViolationCount hysteresis ≠ Regime.nominal := by
  unfold classify
  by_cases hHard : hardViolationCount > 0
  · simp [hHard]
  · simp [hHard, isRecovering]
    by_cases hCritical : meanR < rCritical
    · simp [hCritical]
    · simp [hCritical]

theorem degraded_band_from_nominal_is_degraded
    {meanR hysteresis : Nat}
    (hLower : rCritical <= meanR)
    (hUpper : meanR < rDegraded) :
    classify Regime.nominal meanR 0 hysteresis = Regime.degraded := by
  unfold classify
  simp [Nat.not_lt_of_ge hLower, hUpper, isRecovering]

theorem critical_threshold_from_nominal_is_degraded {hysteresis : Nat} :
    classify Regime.nominal rCritical 0 hysteresis = Regime.degraded := by
  exact degraded_band_from_nominal_is_degraded
    (Nat.le_refl rCritical)
    rCritical_lt_rDegraded

theorem degraded_threshold_from_nominal_is_nominal {hysteresis : Nat} :
    classify Regime.nominal rDegraded 0 hysteresis = Regime.nominal := by
  unfold classify
  simp [Nat.not_lt_of_ge rCritical_le_rDegraded, isRecovering]

theorem degraded_hysteresis_band_stays_degraded
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (hHold : meanR < rDegraded + hysteresis) :
    classify Regime.degraded meanR 0 hysteresis = Regime.degraded := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans rCritical_le_rDegraded hLower)]
  simp [Nat.not_lt_of_ge hLower, hHold]

theorem degraded_past_hysteresis_returns_nominal
    {meanR hysteresis : Nat}
    (hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.degraded meanR 0 hysteresis = Regime.nominal := by
  have hLower : rDegraded <= meanR := by omega
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans rCritical_le_rDegraded hLower)]
  simp [
    Nat.not_lt_of_ge hLower,
    Nat.not_lt_of_ge hPastRecoveryBand,
    isRecovering,
  ]

theorem recovery_degraded_band_stays_recovery
    {meanR hysteresis : Nat}
    (hLower : rCritical <= meanR)
    (hUpper : meanR < rDegraded) :
    classify Regime.recovery meanR 0 hysteresis = Regime.recovery := by
  unfold classify
  simp [Nat.not_lt_of_ge hLower, hUpper, isRecovering]

theorem critical_degraded_band_stays_recovery
    {meanR hysteresis : Nat}
    (hLower : rCritical <= meanR)
    (hUpper : meanR < rDegraded) :
    classify Regime.critical meanR 0 hysteresis = Regime.recovery := by
  unfold classify
  simp [Nat.not_lt_of_ge hLower, hUpper, isRecovering]

theorem critical_high_r_enters_recovery
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (_hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.critical meanR 0 hysteresis = Regime.recovery := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans rCritical_le_rDegraded hLower)]
  simp [Nat.not_lt_of_ge hLower, isRecovering]

theorem critical_degraded_threshold_enters_recovery {hysteresis : Nat} :
    classify Regime.critical rDegraded 0 hysteresis = Regime.recovery := by
  unfold classify
  simp [Nat.not_lt_of_ge rCritical_le_rDegraded, isRecovering]

theorem recovery_high_r_returns_nominal
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.recovery meanR 0 hysteresis = Regime.nominal := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans rCritical_le_rDegraded hLower)]
  simp [Nat.not_lt_of_ge hLower, Nat.not_lt_of_ge hPastRecoveryBand, isRecovering]

theorem recovery_past_hysteresis_returns_nominal
    {meanR hysteresis : Nat}
    (hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.recovery meanR 0 hysteresis = Regime.nominal := by
  have hLower : rDegraded <= meanR := by omega
  exact recovery_high_r_returns_nominal hLower hPastRecoveryBand

theorem hard_violation_is_critical
    {current : Regime} {meanR hardViolationCount hysteresis : Nat}
    (hHard : 0 < hardViolationCount) :
    classify current meanR hardViolationCount hysteresis = Regime.critical := by
  unfold classify
  simp [hHard]

theorem subcritical_r_is_critical
    {current : Regime} {meanR hysteresis : Nat}
    (hLow : meanR < rCritical) :
    classify current meanR 0 hysteresis = Regime.critical := by
  unfold classify
  simp [hLow]

end SPOFormal.Regime
