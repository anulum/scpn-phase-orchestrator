/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Lean projector safety proofs
-/

import Std

set_option autoImplicit false

namespace SPOFormal.Projector

/-- Natural-number clamp used as a fixed-point model of actuator bounds. -/
def clampNat (x lo hi : Nat) : Nat :=
  if x < lo then lo else if hi < x then hi else x

/-- Absolute distance on natural fixed-point actuator units. -/
def absDeltaNat (x y : Nat) : Nat :=
  if x <= y then y - x else x - y

/-- One fixed-point slew step before the final actuator-bound clamp. -/
def limitStepNat (bounded previous limit : Nat) : Nat :=
  clampNat bounded (previous - limit) (previous + limit)

/-- Full fixed-point projection: value bounds first, slew limiting second, final bounds last. -/
def projectFixedNat (proposed previous lo hi limit : Nat) : Nat :=
  let bounded := clampNat proposed lo hi
  clampNat (limitStepNat bounded previous limit) lo hi

theorem clampNat_lower {x lo hi : Nat} (h : lo <= hi) :
    lo <= clampNat x lo hi := by
  unfold clampNat
  split
  · exact Nat.le_refl lo
  · rename_i hNotLow
    split
    · exact h
    · exact Nat.le_of_not_gt hNotLow

theorem clampNat_upper {x lo hi : Nat} (h : lo <= hi) :
    clampNat x lo hi <= hi := by
  unfold clampNat
  split
  · exact h
  · split
    · exact Nat.le_refl hi
    · rename_i hNotLow hNotHigh
      exact Nat.le_of_not_gt hNotHigh

theorem projectFixedNat_lower {proposed previous lo hi limit : Nat} (h : lo <= hi) :
    lo <= projectFixedNat proposed previous lo hi limit := by
  unfold projectFixedNat
  exact clampNat_lower h

theorem projectFixedNat_upper {proposed previous lo hi limit : Nat} (h : lo <= hi) :
    projectFixedNat proposed previous lo hi limit <= hi := by
  unfold projectFixedNat
  exact clampNat_upper h

theorem limitStepNat_delta_le_limit {bounded previous limit : Nat} :
    absDeltaNat (limitStepNat bounded previous limit) previous <= limit := by
  have hWindow : previous - limit <= previous + limit := by
    omega
  have hLower : previous - limit <= limitStepNat bounded previous limit := by
    unfold limitStepNat
    exact clampNat_lower hWindow
  have hUpper : limitStepNat bounded previous limit <= previous + limit := by
    unfold limitStepNat
    exact clampNat_upper hWindow
  unfold absDeltaNat
  split
  · omega
  · omega

theorem limitStepNat_window_lower {bounded previous limit : Nat} :
    previous - limit <= limitStepNat bounded previous limit := by
  unfold limitStepNat
  exact clampNat_lower (by omega)

theorem limitStepNat_window_upper {bounded previous limit : Nat} :
    limitStepNat bounded previous limit <= previous + limit := by
  unfold limitStepNat
  exact clampNat_upper (by omega)

/-- Signed fixed-point clamp, matching the Rust `i32` actuator proof model. -/
def clampInt (x lo hi : Int) : Int :=
  if x < lo then lo else if hi < x then hi else x

/-- Absolute distance on signed fixed-point actuator units. -/
def absDeltaInt (x y : Int) : Int :=
  if x <= y then y - x else x - y

/-- Signed fixed-point slew step before the final actuator-bound clamp. -/
def limitStepInt (bounded previous limit : Int) : Int :=
  if previous + limit < bounded then previous + limit
  else if bounded + limit < previous then previous - limit
  else bounded

/-- Signed fixed-point projection mirroring the Rust projector order. -/
def projectFixedInt (proposed previous lo hi limit : Int) : Int :=
  let bounded := clampInt proposed lo hi
  clampInt (limitStepInt bounded previous limit) lo hi

theorem clampInt_lower {x lo hi : Int} (h : lo <= hi) :
    lo <= clampInt x lo hi := by
  unfold clampInt
  split
  · omega
  · split <;> omega

theorem clampInt_upper {x lo hi : Int} (h : lo <= hi) :
    clampInt x lo hi <= hi := by
  unfold clampInt
  split
  · omega
  · split <;> omega

theorem limitStepInt_delta_le_limit {bounded previous limit : Int}
    (hLimit : 0 <= limit) :
    absDeltaInt (limitStepInt bounded previous limit) previous <= limit := by
  unfold limitStepInt absDeltaInt
  split
  · omega
  · split <;> omega

theorem limitStepInt_window_lower {bounded previous limit : Int}
    (hLimit : 0 <= limit) :
    previous - limit <= limitStepInt bounded previous limit := by
  unfold limitStepInt
  split
  · omega
  · split <;> omega

theorem limitStepInt_window_upper {bounded previous limit : Int}
    (hLimit : 0 <= limit) :
    limitStepInt bounded previous limit <= previous + limit := by
  unfold limitStepInt
  split
  · omega
  · split <;> omega

theorem clampInt_delta_le_limit_of_previous_in_bounds
    {x previous lo hi limit : Int}
    (hLoPrev : lo <= previous)
    (hPrevHi : previous <= hi)
    (hWindowLower : previous - limit <= x)
    (hWindowUpper : x <= previous + limit) :
    absDeltaInt (clampInt x lo hi) previous <= limit := by
  unfold clampInt absDeltaInt
  split
  · omega
  · split <;> omega

theorem projectFixedInt_delta_le_limit
    {proposed previous lo hi limit : Int}
    (hLoPrev : lo <= previous)
    (hPrevHi : previous <= hi)
    (hLimit : 0 <= limit) :
    absDeltaInt (projectFixedInt proposed previous lo hi limit) previous <= limit := by
  unfold projectFixedInt
  exact clampInt_delta_le_limit_of_previous_in_bounds
    hLoPrev
    hPrevHi
    (limitStepInt_window_lower hLimit)
    (limitStepInt_window_upper hLimit)

/-- Adaptive fixed-point rate-limit configuration. -/
structure AdaptiveRateLimitConfig where
  minLimit : Nat
  nominalLimit : Nat
  maxLimit : Nat
  riskGain : Nat
  riskFullScale : Nat

/-- Valid configuration mirrors the Rust `AdaptiveRateLimitConfig::is_valid` contract. -/
def AdaptiveRateLimitConfig.Valid (cfg : AdaptiveRateLimitConfig) : Prop :=
  cfg.minLimit <= cfg.nominalLimit ∧
  cfg.nominalLimit <= cfg.maxLimit ∧
  0 < cfg.riskFullScale

instance (cfg : AdaptiveRateLimitConfig) : Decidable cfg.Valid := by
  unfold AdaptiveRateLimitConfig.Valid
  infer_instance

/-- Integer adaptive-rate formula used for proof-level range reasoning. -/
def adaptiveRateLimitNat (riskSignal : Nat) (cfg : AdaptiveRateLimitConfig) : Nat :=
  if cfg.Valid then
    let saturatedRisk := min riskSignal cfg.riskFullScale
    let adaptiveIncrement := (cfg.riskGain * saturatedRisk) / cfg.riskFullScale
    clampNat (cfg.nominalLimit + adaptiveIncrement) cfg.minLimit cfg.maxLimit
  else
    min cfg.minLimit cfg.maxLimit

theorem adaptiveRateLimitNat_lower {riskSignal : Nat} {cfg : AdaptiveRateLimitConfig}
    (hValid : cfg.Valid) :
    cfg.minLimit <= adaptiveRateLimitNat riskSignal cfg := by
  unfold adaptiveRateLimitNat
  simp [hValid]
  exact clampNat_lower (Nat.le_trans hValid.left hValid.right.left)

theorem adaptiveRateLimitNat_upper {riskSignal : Nat} {cfg : AdaptiveRateLimitConfig}
    (hValid : cfg.Valid) :
    adaptiveRateLimitNat riskSignal cfg <= cfg.maxLimit := by
  unfold adaptiveRateLimitNat
  simp [hValid]
  exact clampNat_upper (Nat.le_trans hValid.left hValid.right.left)

theorem adaptiveRateLimitNat_invalid_fallback
    {riskSignal : Nat} {cfg : AdaptiveRateLimitConfig}
    (hInvalid : ¬ cfg.Valid) :
    adaptiveRateLimitNat riskSignal cfg = min cfg.minLimit cfg.maxLimit := by
  unfold adaptiveRateLimitNat
  simp [hInvalid]

end SPOFormal.Projector
