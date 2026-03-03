# Neuroscience EEG Domainpack

Maps cortical EEG band dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Neural populations oscillate at characteristic frequencies, and inter-region
synchronisation determines cognitive state.  EEG phase-amplitude coupling
*is* Kuramoto dynamics: bandpass → Hilbert → instantaneous phase directly
yields oscillator phase (Buzsaki 2006; Fries 2005 "Communication through
Coherence").

## Layers

| Layer | Oscillators | Channel | Hz Range | Purpose |
|-------|------------|---------|----------|---------|
| delta | 3 | P (hilbert) | 0.5–4 | Deep sleep / pathological sync |
| theta | 2 | P (hilbert) | 4–8 | Memory encoding |
| alpha | 3 | P (hilbert) | 8–13 | Relaxation / flow |
| beta | 2 | P (hilbert) | 13–30 | Arousal / anxiety |
| gamma | 2 | I (event) | 30–100 | Consciousness binding |
| network | 2 | I (event) | < 0.5 | Default mode / salience |

## Boundaries

- **seizure_detection**: broadband_sync <= 0.9 (hard) — Lehnertz (2009)
- **arousal_floor**: alpha_power_ratio >= 0.2 (soft)
- **gamma_ceiling**: gamma_amplitude <= 0.95 (soft)

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-region connectivity |
| lag_delta | alpha | Delta band phase lag |
| entrainment | zeta | External stimulus strength |
| target_phase | Psi | Entrainment target |

## Imprint

Meditation training history (Lutz et al. 2004): repeated sessions accumulate
imprint that modulates coupling and lag, representing long-term plasticity.

## Scenario

300 steps: baseline awake EEG → seizure-like delta hypersynchrony →
alpha entrainment stimulus → meditation imprint accumulation.
