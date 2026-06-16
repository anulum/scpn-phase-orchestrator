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

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time EEG
control checks. It bounds global coupling, delta-band lag, entrainment drive,
and target-phase proposals, then falls back to a zero-stimulus safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
neurostimulation protocol or medical-device control policy.

## Imprint

Meditation training history (Lutz et al. 2004): repeated sessions accumulate
imprint that modulates coupling and lag, representing long-term plasticity.

## Scenario

300 steps: baseline awake EEG → seizure-like delta hypersynchrony →
alpha entrainment stimulus → meditation imprint accumulation.

## Operator Runbook

Use this pack as a research-tier EEG simulation, replay, and policy-review
artefact. It is not a neurostimulation protocol, seizure detector, BCI
controller, or medical-device control path.

### 1. Validate the Binding

```bash
spo validate domainpacks/neuroscience_eeg/binding_spec.yaml
```

Expected result: the binding is valid and reports the `research` safety tier.

### 2. Run the Pack-Owned Scenario

```bash
PYTHONPATH=src python domainpacks/neuroscience_eeg/run.py
```

Expected result: the deterministic EEG scenario executes end to end and prints
a compact JSON summary for review.

### 3. Produce and Verify an Audit Log

```bash
mkdir -p /tmp/spo_neuroscience_eeg_runbook
spo run domainpacks/neuroscience_eeg/binding_spec.yaml \
  --steps 40 \
  --audit /tmp/spo_neuroscience_eeg_runbook/audit.json

spo replay /tmp/spo_neuroscience_eeg_runbook/audit.json --verify
```

Expected result: the research-tier run admits an audit log, and replay
verification passes deterministically.

### 4. Generate Review Artefacts

```bash
spo report /tmp/spo_neuroscience_eeg_runbook/audit.json \
  > /tmp/spo_neuroscience_eeg_runbook/report.txt

spo explain /tmp/spo_neuroscience_eeg_runbook/audit.json \
  --markdown-out /tmp/spo_neuroscience_eeg_runbook/explain.md

spo formal-export domainpacks/neuroscience_eeg/binding_spec.yaml \
  --export policy \
  --output /tmp/spo_neuroscience_eeg_runbook/policy.prism
```

Expected result: the report text, explanation, and policy model are written under
`/tmp/spo_neuroscience_eeg_runbook/`. Full formal packages and STL monitor
exports remain blocked until this binding adds `protocol_net` and
`stl_monitors` evidence.

### 5. Live-Use Readiness Boundary

Do not connect this pack to EEG amplifiers, stimulators, BCI devices, clinical
alarms, or patient workflows from the local runbook. A live research or
clinical deployment needs, at minimum:

- subject/data-governance approval and consent path for the intended dataset;
- hardware adapter provenance, isolation, and latency evidence;
- admitted audit-log environment with replay/report/explain retention;
- externally reviewed policy and monitor evidence for the intended protocol;
- operator approval, version pinning, and rollback procedure;
- explicit statement that SPO does not diagnose, detect, or treat neurological
  conditions.
