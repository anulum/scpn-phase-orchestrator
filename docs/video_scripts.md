<!--
SPDX-License-Identifier: AGPL-3.0-or-later
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-->

# Video Scripts (60-Second Demos)

Record these with Loom, OBS, or any screen recorder. Each script is
designed for a 60-second video with terminal output. Pin in README.

---

## Roadmap Walkthrough Set (v1.0 Adoption Track)

These five scripts match the roadmap onboarding targets exactly:

1. first run
2. binding-spec authoring
3. policy debugging
4. audit replay
5. deployment profiles

### A. "First Run" (60s)

**Screen:** Terminal

```bash
pip install scpn-phase-orchestrator
spo validate domainpacks/minimal_domain/binding_spec.yaml
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 120 --audit first_run_audit.jsonl --seed 42
spo report first_run_audit.jsonl
```

Voiceover points:
- install
- validate
- run
- report from the same audit trace

### B. "Binding-Spec Authoring" (60s)

**Screen:** Editor + terminal

```bash
spo scaffold my_domain
$EDITOR domainpacks/my_domain/binding_spec.yaml
spo validate domainpacks/my_domain/binding_spec.yaml
```

Voiceover points:
- map raw sources into P/I/S oscillator families
- define layers/objectives/actuators
- validate after each edit

### C. "Policy Debugging" (60s)

**Screen:** Editor + terminal

```bash
$EDITOR domainpacks/my_domain/policy.yaml
spo run domainpacks/my_domain/binding_spec.yaml --steps 200 --audit policy_debug_audit.jsonl --seed 7
spo report policy_debug_audit.jsonl
```

Voiceover points:
- show one threshold/action rule
- run with audit
- verify actions/regimes in report output

### D. "Audit Replay" (60s)

**Screen:** Terminal

```bash
spo replay policy_debug_audit.jsonl --verify
spo report policy_debug_audit.jsonl
```

Voiceover points:
- deterministic replay
- hash-chain verification
- identical run evidence for review and incident analysis

### E. "Deployment Profiles" (60s)

**Screen:** Terminal + docs page

```bash
python -m scpn_phase_orchestrator.cli --help
python -c "import spo_kernel; print('spo_kernel OK')"
python -c "from scpn_phase_orchestrator.nn import HAS_JAX; print(HAS_JAX)"
python -m pytest -q tests/test_backend_module_imports.py
```

Voiceover points:
- Python-only, Rust FFI, JAX, and full profiles
- each profile has an explicit preflight command
- expected fallback behavior is documented

---

## 1. "What is SPO?" (60s)

**Screen:** Terminal

```
[TITLE CARD: "SCPN Phase Orchestrator — 60 Second Demo"]

> pip install scpn-phase-orchestrator
> spo demo --domain plasma_control --steps 100

[Show output scrolling: R values climbing, regime transitions]

VOICEOVER:
"SPO is a domain-agnostic coherence control compiler. You give it
a YAML binding spec — describing your domain — and it synchronises
coupled oscillators with closed-loop supervisory control.

This is plasma control: 16 oscillators modelling MHD modes in a
tokamak. The supervisor detects desynchronisation and boosts
coupling automatically.

No other library does this. Install with pip, run with one command."

> spo demo --domain cardiac_rhythm --steps 100

"Same engine, different physics. Cardiac rhythm: 10 oscillators,
4 layers, SA node pacemaker model. Same API."

[END CARD: "pip install scpn-phase-orchestrator | github.com/anulum"]
```

---

## 2. "Supervisor Advantage" (60s)

**Screen:** Terminal running `python examples/supervisor_advantage.py`

```
VOICEOVER:
"Every other Kuramoto library is open-loop: simulate and observe.
SPO adds closed-loop control — detect, decide, act.

This example runs the same 8 oscillators twice: once passive,
once with SPO's supervisor monitoring R and boosting coupling
when it drops below 0.4.

[Show the comparison table printing]

The supervised version achieves higher coherence — that's the
difference between a simulator and a controller.

This is why fusion labs, power grid operators, and neuroscience
researchers need SPO: real-time intervention, not just observation."
```

---

## 3. "Failure Recovery" (60s)

**Screen:** Terminal running `python examples/failure_recovery.py`

```
VOICEOVER:
"Phase 1: 8 oscillators synchronise normally. R reaches 0.9.

Phase 2: We disconnect 3 nodes — simulating a sensor failure,
a tripped generator, or a lost network link. R drops immediately.

Phase 3: The supervisor detects the degradation and triples the
coupling on remaining links. R recovers.

This pattern — detect fault, compensate, recover — is what makes
SPO production-ready for safety-critical systems. Power grids,
tokamaks, medical devices."
```

---

## 4. "32 Domains, One Engine" (60s)

**Screen:** Terminal running `python examples/cross_domain_universality.py`

```
VOICEOVER:
"Same 4 lines of code. Five completely different domains.

Plasma control: 16 oscillators, MHD mode coupling.
Cardiac rhythm: SA node pacemaker.
Power grid: inertial swing equation.
Traffic flow: green wave synchronisation.
Neuroscience: EEG alpha-band coherence.

The binding spec YAML is the only thing that changes. The engine,
supervisor, monitors, and actuation layer are universal.

33 domainpacks ship out of the box. Create your own with
spo scaffold."
```

---

## 5. "From N=4 to N=1000" (60s)

**Screen:** Terminal running `python examples/scaling_showcase.py`

```
VOICEOVER:
"SPO scales from toy examples to production.

4 oscillators: microseconds per step.
64 oscillators: still fast.
256: sub-millisecond with the Rust kernel.
1000: production scale, no code changes.

Same API. Same supervisor. Only N changes.

For 10,000+ oscillators, add the JAX GPU backend.
For sub-microsecond latency, there's the FPGA kernel."
```

---

## 6. "Learn the Physics" (60s)

**Screen:** Terminal running `python examples/inverse_coupling_demo.py`

```
VOICEOVER:
"The inverse problem: you observe phase trajectories, but you
don't know the coupling matrix.

SPO solves this with gradient descent on the synchronisation
cost. Start with a uniform guess, compute the gradient via
finite differences, and iterate.

After 20 iterations, the learned matrix converges toward the
ground truth. For GPU acceleration, use the JAX nn/ module
with automatic differentiation."
```

---

## 7. "EEG to Phase Dynamics" (60s)

**Screen:** Terminal running `python examples/eeg_file_ingestion.py`

```
VOICEOVER:
"Real data in, coherence analysis out.

Step 1: Load EEG signals — 8 electrodes, alpha band.
Step 2: Extract phases via Hilbert transform.
Step 3: Build coupling from electrode distances.
Step 4: Run SPO with chimera detection and NPE monitoring.

Replace the synthetic data with your own EEG files — the pipeline
is the same. MNE, NumPy, pandas — any loader works."
```

---

## Recording Tips

- Terminal font: 16pt monospace, dark background
- Record at 1080p, 30fps
- Keep terminal output visible (no scrolling past important lines)
- Add title/end cards in post-production
- Upload to YouTube, embed in README via thumbnail + link
- Pin the "What is SPO?" video at the top of README
