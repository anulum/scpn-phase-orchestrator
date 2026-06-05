/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Lean continuous-envelope proofs
-/

import SPOFormal.Kinematic

set_option autoImplicit false

namespace SPOFormal.Continuous

open SPOFormal.Kinematic

/-- Fixed-point continuous-time envelope for PHA-C moving-frame handoffs.

Rates are measured in metric fixed-point units per second. Time is measured in
the caller's fixed-point clock units. This keeps the proof dependency-light
while still making the continuous horizon explicit before the runtime projects
the envelope onto discrete integration steps.
-/
structure ContinuousEnvelopeBounds where
  initialTolerance : Nat
  relativeVelocityRateBound : Nat
  couplingResidualRateBound : Nat
  timeScaleUnitsPerSecond : Nat
  horizonTimeUnits : Nat
  mergeWindowTolerance : Nat
  deriving DecidableEq, Repr

/-- Continuous additive drive rate: relative velocity plus coupling residual. -/
def ContinuousEnvelopeBounds.driveRateBound
    (cfg : ContinuousEnvelopeBounds) : Nat :=
  cfg.relativeVelocityRateBound + cfg.couplingResidualRateBound

/-- Sample a per-second drive rate over an arbitrary fixed-point time horizon. -/
def ContinuousEnvelopeBounds.sampledDriveBoundAt
    (cfg : ContinuousEnvelopeBounds)
    (timeUnits : Nat) : Nat :=
  ceilDiv
    (cfg.driveRateBound * timeUnits)
    cfg.timeScaleUnitsPerSecond

/-- Additive continuous-envelope budget at a fixed-point time horizon. -/
def ContinuousEnvelopeBounds.budgetAt
    (cfg : ContinuousEnvelopeBounds)
    (timeUnits : Nat) : Nat :=
  cfg.initialTolerance + cfg.sampledDriveBoundAt timeUnits

/-- Runtime-facing continuous-envelope certificate at the reviewed horizon. -/
def ContinuousEnvelopeBounds.budgetCertificate
    (cfg : ContinuousEnvelopeBounds) : Bool :=
  cfg.budgetAt cfg.horizonTimeUnits <= cfg.mergeWindowTolerance

/-- Boolean reflection for the continuous-envelope horizon certificate. -/
theorem budgetCertificate_eq_true_iff
    {cfg : ContinuousEnvelopeBounds} :
    cfg.budgetCertificate = true ↔
      cfg.budgetAt cfg.horizonTimeUnits <= cfg.mergeWindowTolerance := by
  unfold ContinuousEnvelopeBounds.budgetCertificate
  simp

/-- A true continuous-envelope certificate discharges the reviewed horizon. -/
theorem continuous_envelope_certificate_discharges_horizon
    {cfg : ContinuousEnvelopeBounds}
    (hCertificate : cfg.budgetCertificate = true) :
    cfg.budgetAt cfg.horizonTimeUnits <= cfg.mergeWindowTolerance := by
  exact (budgetCertificate_eq_true_iff (cfg := cfg)).mp hCertificate

/-- If a caller has a monotone reviewed budget table, the horizon certificate
discharges every fixed-point time prefix without importing an analysis stack. -/
theorem continuous_envelope_certificate_discharges_prefixes
    {cfg : ContinuousEnvelopeBounds}
    (hCertificate : cfg.budgetCertificate = true)
    (hBudgetMonotone : ∀ timeUnits,
      timeUnits <= cfg.horizonTimeUnits ->
        cfg.budgetAt timeUnits <= cfg.budgetAt cfg.horizonTimeUnits) :
    ∀ timeUnits, timeUnits <= cfg.horizonTimeUnits ->
      cfg.budgetAt timeUnits <= cfg.mergeWindowTolerance := by
  intro timeUnits hTime
  exact Nat.le_trans
    (hBudgetMonotone timeUnits hTime)
    (continuous_envelope_certificate_discharges_horizon
      (cfg := cfg)
      hCertificate)

/-- Convert a continuous-rate envelope into the sampled discrete kinematic
certificate used by runtime PHA-C manifests. -/
def ContinuousEnvelopeBounds.toSampledRateKinematicBounds
    (cfg : ContinuousEnvelopeBounds)
    (lipschitzStepGain stepTimeUnits horizonSteps : Nat) :
    SampledRateKinematicBounds := {
  initialTolerance := cfg.initialTolerance
  lipschitzStepGain := lipschitzStepGain
  relativeVelocityRateBound := cfg.relativeVelocityRateBound
  couplingResidualRateBound := cfg.couplingResidualRateBound
  timeScaleUnitsPerSecond := cfg.timeScaleUnitsPerSecond
  stepTimeUnits := stepTimeUnits
  mergeWindowTolerance := cfg.mergeWindowTolerance
  horizonSteps := horizonSteps
}

/-- The continuous envelope can feed the existing sampled-rate discrete budget
once the runtime supplies the integration step size and step horizon. -/
theorem continuous_envelope_sampled_step_certificate_discharges_budget
    {cfg : ContinuousEnvelopeBounds}
    {lipschitzStepGain stepTimeUnits horizonSteps : Nat}
    (hCertificate :
      (cfg.toSampledRateKinematicBounds
        lipschitzStepGain
        stepTimeUnits
        horizonSteps).budgetCertificate = true) :
    ∀ k, k <= horizonSteps ->
      (cfg.toSampledRateKinematicBounds
        lipschitzStepGain
        stepTimeUnits
        horizonSteps).toKinematicBounds.budget k <=
          cfg.mergeWindowTolerance := by
  exact sampled_rate_certificate_discharges_budget
    (cfg := cfg.toSampledRateKinematicBounds
      lipschitzStepGain
      stepTimeUnits
      horizonSteps)
    hCertificate

end SPOFormal.Continuous
