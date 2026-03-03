# Domain Utilisation Schemas

Cross-domain comparison of all SPO domainpacks, showing how Kuramoto/UPDE
phase dynamics map to diverse physical, biological, and engineered systems.

## Master Domainpack Table

| Pack | Layers | Osc | Safety Tier | Key Innovation |
|------|--------|-----|-------------|---------------|
| bio_stub | 3 | 7 | research | Placeholder biology template |
| fusion_equilibrium | 4 | 10 | research | MHD equilibrium + transport |
| geometry_walk | 2 | 6 | research | Random walk on graphs |
| manufacturing_spc | 3 | 9 | consumer | SPC process drift detection |
| metaphysics_demo | 3 | 7 | research | P/I/S + imprint + geometry |
| minimal_domain | 2 | 4 | research | Minimal valid binding spec |
| plasma_control | 8 | 16 | research | Full tokamak layer hierarchy |
| quantum_simulation | 3 | 8 | research | Quantum gate phase tracking |
| queuewaves | 3 | 6 | consumer | Service queue oscillations |
| neuroscience_eeg | 6 | 14 | research | EEG band→phase, seizure detection |
| power_grid | 5 | 12 | production | Swing equation = Kuramoto (exact) |
| cardiac_rhythm | 4 | 10 | clinical | Gap-junction coupling, arrhythmia |
| rotating_machinery | 4 | 10 | consumer | Vibration harmonics, ISO 10816 |
| chemical_reactor | 4 | 10 | production | Hopf bifurcation, Semenov limit |

## Why Kuramoto Fits Each Domain

### Neuroscience (EEG)

Neural populations oscillate at characteristic band frequencies.  Bandpass
filtering → Hilbert transform yields instantaneous phase, which IS a
Kuramoto oscillator phase.  Inter-region phase-locking value (PLV) measures
synchronisation.  Buzsaki (2006) *Rhythms of the Brain*; Fries (2005)
"Communication through Coherence".

### Power Systems

The swing equation `dδ/dt = ω` is literally a second-order Kuramoto model.
PMU phasor angles are oscillator phases; line admittances are coupling
constants.  No phase extraction step is needed — measurement IS phase.
Dorfler, Chertkov, Bullo (2013).

### Cardiac Electrophysiology

Gap-junction (connexin-43) electrical coupling between cardiac cells is
Kuramoto coupling with coupling constant proportional to gap junction
conductance.  SA node pacemaker cells entrain downstream tissue exactly
as high-frequency Kuramoto oscillators entrain slower ones.  Strogatz (2003)
*Sync*.

### Rotating Machinery

Vibration harmonics (1X, 2X, 3X) of shaft rotation, bearing defect
frequencies (BPFI, BPFO, FTF), and structural resonance modes form a
coupled oscillator network linked by mechanical impedance.  Phase
relationships between harmonics diagnose faults: 1X+2X in-phase signals
misalignment.  ISO 10816-3.

### Chemical Reactors

CSTR systems undergo Hopf bifurcations where concentration and temperature
oscillate with well-defined phase relationships.  The Arrhenius-heat
coupling creates limit cycles naturally modelled as coupled oscillators.
Fogler (2020) Ch. 12.

### Plasma Physics

Micro-turbulence, zonal flows, MHD tearing modes, sawteeth/ELMs, and
transport barriers form a multi-timescale oscillator hierarchy.  The
predator-prey relationship between turbulence and zonal flows is a
classic coupled-oscillator problem.  ITER Physics Basis (2007).

### Manufacturing SPC

Sensor signals (vibration, temperature, pressure) oscillate around
setpoints.  Tool wear causes systematic drift that correlates sensor
phases — exactly the kind of synchronisation Kuramoto detects.

### Queue Networks

Service queues exhibit oscillatory behaviour under periodic demand.
Phase relationships between upstream and downstream queues determine
system throughput.  Spiked arrivals synchronise queue oscillations.

## Phase Extraction Rationale

| Domain | Source Signal | Extraction | Phase = |
|--------|-------------|-----------|---------|
| EEG | Voltage time series | Bandpass + Hilbert | Instantaneous phase |
| Power grid | PMU phasor | Direct measurement | Rotor angle δ |
| Cardiac | Intracardiac EGM | Activation time mapping | Activation phase |
| Rotating | Accelerometer | Order tracking + FFT | Harmonic phase |
| Chemical | T, C, P sensors | Detrend + Hilbert | Oscillation phase |
| Plasma | Mirnov coils, ECE | Mode fitting | Mode phase |
| Manufacturing | SPC sensor | Detrend around setpoint | Deviation phase |
| Queue | Queue depth | Detrend + Hilbert | Demand phase |

## Good/Bad Layer Partition

| Domain | Good (R↑ = healthy) | Bad (R↑ = pathological) |
|--------|--------------------|-----------------------|
| EEG | Alpha, gamma, network | Delta (wake), beta excess |
| Power grid | Generator sync, area freq | Load demand, renewable |
| Cardiac | SA node, atrial | Ventricular desync |
| Rotating | Shaft at nominal | Blade flutter, structural |
| Chemical | Heat transfer, feed flow | Kinetics oscillation |
| Plasma | Transport barrier, current | Turbulence, sawteeth |
| Manufacturing | Machine, line | Sensor (drift = bad) |
| Queue | Server, throughput | Arrival burst |

## Boundary Sources

| Domain | Hard Boundaries | Standard/Source |
|--------|----------------|----------------|
| EEG | Broadband sync < 0.9 | Lehnertz (2009) |
| Power grid | Freq ±0.5 Hz, V 0.95–1.05 pu | NERC BAL-003-2, ANSI C84.1 |
| Cardiac | HR 40–180 bpm, QT < 500 ms | ACC/AHA guidelines, Roden (2004) |
| Rotating | Vibration < 7.1 mm/s | ISO 10816-3 zone C/D |
| Chemical | T < 450°C, P < 15 bar | Semenov limit, ASME VIII |
| Plasma | q_min >= 1, β_N <= 2.8 | Kruskal-Shafranov, Troyon |
| Manufacturing | Temp < 85°C, pressure > 2 | OEM specs |

## Actuator Mapping

| Domain | K (coupling) | alpha (lag) | zeta (drive) | Psi (target) |
|--------|-------------|------------|-------------|-------------|
| EEG | Connectivity | Delta band lag | Entrainment stim | Target phase |
| Power grid | Governor droop | Load shed phase | AGC bias | Curtailment |
| Cardiac | Drug coupling | Vagal modulation | Pacing rate | Pacing target |
| Rotating | Bearing stiffness | Damper viscosity | Speed setpoint | — |
| Chemical | Coolant flow | Agitator speed | Feed rate | Jacket SP |
| Plasma | Global coupling | Turbulence lag | Damping | — |
| Manufacturing | Global coupling | Sensor lag | Damping | — |

## Coupling Topology Rationale

| Domain | Topology | Rationale |
|--------|---------|-----------|
| EEG | Symmetric, non-negative | Cortical connectivity is undirected |
| Power grid | Distance-decayed | Admittance ~ 1/impedance ~ 1/distance |
| Cardiac | Strong nearest-neighbour | Gap junctions connect adjacent cells |
| Rotating | Layer-block | Mechanical path: shaft → bearing → structure |
| Chemical | Dense intra-layer | Heat-mass coupling is tight within reactions |
| Plasma | Hierarchical decay | Timescale separation between layers |
| Manufacturing | Weak cross-layer | Sensors are only indirectly coupled |

## Imprint Semantics

| Domain | Physical Meaning | Timescale | Modulates |
|--------|-----------------|-----------|-----------|
| EEG | Meditation training (plasticity) | Weeks–months | K, alpha |
| Cardiac | Drug accumulation (pharmacokinetics) | Hours–days | K |
| Chemical | Catalyst fouling/deactivation | Months | K |
| Manufacturing | Tool wear history | Weeks | K |
| Plasma | (none) | — | — |
| Power grid | (none) | — | — |
| Rotating | (none) | — | — |

## Adding a New Domain

1. Identify the oscillators: what physically oscillates?
2. Map oscillators to layers by timescale hierarchy.
3. Define omega values from domain physics (cite sources).
4. Partition layers into good (sync = healthy) and bad (sync = pathological).
5. Set boundaries from engineering standards or medical guidelines.
6. Map actuators to physical control knobs.
7. Decide if imprint applies (slow accumulation effects).
8. Create `binding_spec.yaml`, `policy.yaml`, `run.py`, `README.md`.
9. Validate: `spo validate domainpacks/<name>/binding_spec.yaml`.
10. Run: `python domainpacks/<name>/run.py`.
