# Circadian Biology Domainpack

Maps circadian rhythm dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

SCN (suprachiasmatic nucleus) neurons are literal coupled oscillators with
~24 h period.  Clock genes (Per/Cry, Bmal1, Rev-erb) form transcription-
translation feedback loops whose phase relationships determine circadian
entrainment.  Winfree (1967) first modelled biological clocks as coupled
oscillators; Strogatz (2003) "Sync" Ch. 5 formalises SCN synchronisation.

## Layers

| Layer | Oscillators | Channel | Period | Purpose |
|-------|------------|---------|--------|---------|
| scn_core | 3 | P (hilbert) | ~24 h | Per/Cry, Bmal1, Rev-erb clock genes |
| peripheral_clocks | 3 | P (hilbert) | ~24 h | Liver, kidney, adipose tissue clocks |
| metabolic_rhythm | 2 | I (event) | ~12-24 h | Glucose cycling, cortisol rhythm |
| behavioral | 2 | I (event) | ~24 h | Sleep/wake, feeding schedule |

## Boundaries

- **phase_deviation**: < 3 h (hard) -- clinical circadian disruption threshold
- **cortisol_acrophase**: < 2 h shift (soft) -- HPA axis dysregulation
- **temperature_nadir**: < 1 h shift (soft) -- core body temperature marker

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| light_exposure | zeta | Zeitgeber drive strength |
| meal_timing | Psi | Peripheral clock phase target |
| coupling_strength | K | Inter-clock coupling |
| sleep_schedule | alpha | Behavioral phase lag |

## Imprint

Chronic jet lag / shift work debt: repeated circadian disruption accumulates
imprint that weakens coupling and shifts phase lags (Czeisler et al. 1999).

## Scenario

250 steps: entrained rhythm -> jet lag (SCN phase shift) -> peripheral
free-running -> light therapy intervention -> re-entrainment.
