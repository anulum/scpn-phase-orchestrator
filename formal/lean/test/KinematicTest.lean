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

set_option autoImplicit false

namespace SPOFormal.KinematicTest

open SPOFormal.Kinematic

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

end SPOFormal.KinematicTest
