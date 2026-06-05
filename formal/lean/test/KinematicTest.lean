/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Lean kinematic smoke tests
-/

import SPOFormal.Kinematic
import SPOFormal.Continuous

set_option autoImplicit false

namespace SPOFormal.KinematicTest

open SPOFormal.Kinematic
open SPOFormal.Continuous

def mifSmokeBounds : KinematicBounds := {
  initialTolerance := 2
  lipschitzStepGain := 0
  relativeVelocityStepBound := 1
  couplingResidualStepBound := 1
  mergeWindowTolerance := 10
  horizonSteps := 4
}

example : mifSmokeBounds.driveBound = 2 := by
  decide

example : mifSmokeBounds.budget 4 = 10 := by
  decide

example :
    ∀ k, k <= mifSmokeBounds.horizonSteps ->
      mifSmokeBounds.budget k <= mifSmokeBounds.mergeWindowTolerance := by
  exact zero_gain_budget_within_horizon (cfg := mifSmokeBounds) rfl (by decide)

example : mifSmokeBounds.zeroGainCertificate = true := by
  decide

example :
    ∀ k, k <= mifSmokeBounds.horizonSteps ->
      mifSmokeBounds.budget k <= mifSmokeBounds.mergeWindowTolerance := by
  exact zero_gain_certificate_discharges_budget
    (cfg := mifSmokeBounds)
    (by decide)

example : mifSmokeBounds.budgetCertificate = true := by
  decide

def mifGainSmokeBounds : KinematicBounds := {
  initialTolerance := 1
  lipschitzStepGain := 1
  relativeVelocityStepBound := 1
  couplingResidualStepBound := 0
  mergeWindowTolerance := 7
  horizonSteps := 2
}

example : mifGainSmokeBounds.budget 2 = 7 := by
  decide

example : mifGainSmokeBounds.budgetCertificate = true := by
  decide

example :
    ∀ k, k <= mifGainSmokeBounds.horizonSteps ->
      mifGainSmokeBounds.budget k <= mifGainSmokeBounds.mergeWindowTolerance := by
  exact budget_certificate_discharges_budget
    (cfg := mifGainSmokeBounds)
    (by decide)

def mifSampledRateBounds : SampledRateKinematicBounds := {
  initialTolerance := 1
  lipschitzStepGain := 0
  relativeVelocityRateBound := 5
  couplingResidualRateBound := 0
  timeScaleUnitsPerSecond := 10
  stepTimeUnits := 2
  mergeWindowTolerance := 4
  horizonSteps := 3
}

example : mifSampledRateBounds.sampledStepBound 5 = 1 := by
  decide

example : mifSampledRateBounds.toKinematicBounds.driveBound = 1 := by
  decide

example : mifSampledRateBounds.budgetCertificate = true := by
  decide

example :
    ∀ k, k <= mifSampledRateBounds.horizonSteps ->
      mifSampledRateBounds.toKinematicBounds.budget k <=
        mifSampledRateBounds.mergeWindowTolerance := by
  exact sampled_rate_certificate_discharges_budget
    (cfg := mifSampledRateBounds)
    (by decide)

def mifPhaseBudgetBounds : PhaseBudgetBounds := {
  maxPhaseDispersion := 2
  configuredPhaseDrift := 1
  phaseTolerance := 4
}

def mifAcceptanceReplayBounds : AcceptanceKinematicReplayBounds := {
  equationsValidated := true
  summaryReplayTolerance := 1
  summaryReplayToleranceLimit := 1
}

example : mifPhaseBudgetBounds.phaseBudget = 3 := by
  decide

example : mifPhaseBudgetBounds.budgetCertificate = true := by
  decide

example :
    mifPhaseBudgetBounds.phaseBudget <= mifPhaseBudgetBounds.phaseTolerance := by
  exact phase_budget_certificate_discharges_phase_lock
    (cfg := mifPhaseBudgetBounds)
    (by decide)

example : mifAcceptanceReplayBounds.replayCertificate = true := by
  decide

example :
    mifAcceptanceReplayBounds.equationsValidated = true ∧
      mifAcceptanceReplayBounds.summaryReplayTolerance <=
        mifAcceptanceReplayBounds.summaryReplayToleranceLimit := by
  exact acceptance_replay_certificate_discharges_runtime_preconditions
    (cfg := mifAcceptanceReplayBounds)
    (by decide)

example :
    mifSmokeBounds.acceptanceCertificate
      mifPhaseBudgetBounds
      mifAcceptanceReplayBounds = true := by
  decide

example :
    mifSmokeBounds.budget mifSmokeBounds.horizonSteps <=
      mifSmokeBounds.mergeWindowTolerance ∧
        mifPhaseBudgetBounds.phaseBudget <=
          mifPhaseBudgetBounds.phaseTolerance ∧
            mifAcceptanceReplayBounds.equationsValidated = true ∧
              mifAcceptanceReplayBounds.summaryReplayTolerance <=
                mifAcceptanceReplayBounds.summaryReplayToleranceLimit := by
  exact acceptance_certificate_discharges_runtime_preconditions
    (kinCfg := mifSmokeBounds)
    (phaseCfg := mifPhaseBudgetBounds)
    (replayCfg := mifAcceptanceReplayBounds)
    (by decide)

example (distance : Nat -> Nat)
    (hInitial : distance 0 <= mifSmokeBounds.initialTolerance)
    (hStep : ∀ k, distance (k + 1) <= distance k + mifSmokeBounds.driveBound) :
    ∀ k, k <= mifSmokeBounds.horizonSteps ->
      distance k <= mifSmokeBounds.mergeWindowTolerance := by
  exact merge_window_invariant_zero_gain
    distance
    mifSmokeBounds
    rfl
    (by decide)
    hInitial
    hStep

example : mergeWindowLocked 7 8 10 10 = true := by
  decide

example : mergeWindowLocked 9 8 10 10 = false := by
  decide

example :
    mergeWindowLocked 7 8 10 10 = true ↔ 7 <= 8 ∧ 10 <= 10 := by
  exact mergeWindowLocked_eq_true_iff

example : mergeWindowLockedWithPhaseBudget mifPhaseBudgetBounds 10 10 = true := by
  decide

example :
    mergeWindowLockedWithPhaseBudget mifPhaseBudgetBounds 11 10 = false := by
  decide

example :
    mergeWindowLockedWithPhaseBudget mifPhaseBudgetBounds 10 10 = true ↔
      mifPhaseBudgetBounds.phaseBudget <= mifPhaseBudgetBounds.phaseTolerance ∧
        10 <= 10 := by
  exact mergeWindowLockedWithPhaseBudget_eq_true_iff

example (spatialDistance : Nat -> Nat)
    (hSpatial : ∀ k, k <= mifSmokeBounds.horizonSteps ->
      spatialDistance k <= mifSmokeBounds.mergeWindowTolerance) :
    ∀ k, k <= mifSmokeBounds.horizonSteps ->
      mergeWindowLockedWithPhaseBudget
        mifPhaseBudgetBounds
        (spatialDistance k)
        mifSmokeBounds.mergeWindowTolerance = true := by
  exact merge_window_locked_with_phase_budget_over_horizon
    spatialDistance
    mifPhaseBudgetBounds
    mifSmokeBounds
    hSpatial
    (by decide)

def mifContinuousEnvelopeBounds : ContinuousEnvelopeBounds := {
  initialTolerance := 1
  relativeVelocityRateBound := 5
  couplingResidualRateBound := 0
  timeScaleUnitsPerSecond := 10
  horizonTimeUnits := 6
  mergeWindowTolerance := 4
}

example : mifContinuousEnvelopeBounds.driveRateBound = 5 := by
  decide

example : mifContinuousEnvelopeBounds.sampledDriveBoundAt 6 = 3 := by
  decide

example : mifContinuousEnvelopeBounds.budgetAt 6 = 4 := by
  decide

example : mifContinuousEnvelopeBounds.budgetCertificate = true := by
  decide

example :
    mifContinuousEnvelopeBounds.budgetAt
      mifContinuousEnvelopeBounds.horizonTimeUnits <=
        mifContinuousEnvelopeBounds.mergeWindowTolerance := by
  exact continuous_envelope_certificate_discharges_horizon
    (cfg := mifContinuousEnvelopeBounds)
    (by decide)

example :
    (mifContinuousEnvelopeBounds.toSampledRateKinematicBounds
      0
      2
      3).toKinematicBounds.driveBound = 1 := by
  decide

example :
    ∀ k, k <= 3 ->
      (mifContinuousEnvelopeBounds.toSampledRateKinematicBounds
        0
        2
        3).toKinematicBounds.budget k <=
          mifContinuousEnvelopeBounds.mergeWindowTolerance := by
  exact continuous_envelope_sampled_step_certificate_discharges_budget
    (cfg := mifContinuousEnvelopeBounds)
    (lipschitzStepGain := 0)
    (stepTimeUnits := 2)
    (horizonSteps := 3)
    (by decide)

end SPOFormal.KinematicTest
