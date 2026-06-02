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

/-- Recovering regimes must pass through recovery before returning nominal. -/
def isRecovering : Regime -> Bool
  | Regime.critical => true
  | Regime.recovery => true
  | _ => false

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

theorem nominal_safe_summary_never_classifies_critical
    {meanR hysteresis : Nat} (hMean : rCritical <= meanR) :
    classify Regime.nominal meanR 0 hysteresis ≠ Regime.critical := by
  unfold classify
  simp [Nat.not_lt_of_ge hMean, isRecovering]
  split <;> simp

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

theorem degraded_hysteresis_band_stays_degraded
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (hHold : meanR < rDegraded + hysteresis) :
    classify Regime.degraded meanR 0 hysteresis = Regime.degraded := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans (by decide : rCritical <= rDegraded) hLower)]
  simp [Nat.not_lt_of_ge hLower, hHold]

theorem critical_high_r_enters_recovery
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (_hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.critical meanR 0 hysteresis = Regime.recovery := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans (by decide : rCritical <= rDegraded) hLower)]
  simp [Nat.not_lt_of_ge hLower, isRecovering]

theorem recovery_high_r_returns_nominal
    {meanR hysteresis : Nat}
    (hLower : rDegraded <= meanR)
    (hPastRecoveryBand : rDegraded + hysteresis <= meanR) :
    classify Regime.recovery meanR 0 hysteresis = Regime.nominal := by
  unfold classify
  simp [Nat.not_lt_of_ge (Nat.le_trans (by decide : rCritical <= rDegraded) hLower)]
  simp [Nat.not_lt_of_ge hLower, Nat.not_lt_of_ge hPastRecoveryBand, isRecovering]

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
