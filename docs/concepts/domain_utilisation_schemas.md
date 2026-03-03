# Domain Utilisation Schemas

Cross-domain comparison of all 21 SPO domainpacks, showing how Kuramoto/UPDE
phase dynamics map to diverse physical, biological, and engineered systems.

## Master Domainpack Table

| Pack | Layers | Osc | Safety Tier | Pipeline | Key Innovation |
|------|--------|-----|-------------|----------|---------------|
| bio_stub | 4 | 16 | research | full | Multi-scale biological oscillators |
| cardiac_rhythm | 4 | 10 | clinical | full | Gap-junction coupling, arrhythmia |
| chemical_reactor | 4 | 10 | production | full | Hopf bifurcation, Semenov limit |
| circadian_biology | 4 | 10 | research | full | SCN clock-gene coupled oscillators |
| epidemic_sir | 3 | 8 | research | full | Epidemic wave synchronisation |
| firefly_swarm | 2 | 8 | research | full | Mirollo-Strogatz flash synchronisation |
| fusion_equilibrium | 6 | 12 | research | full | MHD equilibrium + FusionCoreBridge |
| geometry_walk | 2 | 8 | research | full | Random walk on graphs |
| laser_array | 3 | 8 | research | full | Evanescent-coupled laser phase-locking |
| manufacturing_spc | 3 | 9 | consumer | full | SPC process drift detection |
| metaphysics_demo | 3 | 7 | research | full | P/I/S + imprint + geometry |
| minimal_domain | 2 | 4 | research | full | Minimal-but-complete pipeline example |
| neuroscience_eeg | 6 | 14 | research | full | EEG band->phase, seizure detection |
| plasma_control | 8 | 16 | research | adapter | Full tokamak layer hierarchy (PlasmaControlBridge) |
| pll_clock | 3 | 8 | production | full | PLL network clock sync (ITU-T G.811) |
| power_grid | 5 | 12 | production | full | Swing equation = Kuramoto (exact) |
| quantum_simulation | 3 | 8 | research | adapter | Quantum gate phase tracking (QuantumControlBridge) |
| queuewaves | 3 | 6 | consumer | full | Service queue oscillations |
| rotating_machinery | 4 | 10 | consumer | full | Vibration harmonics, ISO 10816 |
| swarm_robotics | 3 | 8 | consumer | full | Vicsek collective motion/formation |
| traffic_flow | 4 | 10 | consumer | full | Signal coordination = phase sync |

**Pipeline types**: *full* = BoundaryObserver + RegimeManager + SupervisorPolicy + PolicyEngine + ImprintModel (where applicable).  *adapter* = uses a specialised bridge class (FusionCoreBridge, PlasmaControlBridge, QuantumControlBridge) as an alternative architecture.

## Why Kuramoto Fits Each Domain

### Neuroscience (EEG)

Neural populations oscillate at characteristic band frequencies.  Bandpass
filtering -> Hilbert transform yields instantaneous phase, which IS a
Kuramoto oscillator phase.  Inter-region phase-locking value (PLV) measures
synchronisation.  Buzsaki (2006) *Rhythms of the Brain*; Fries (2005)
"Communication through Coherence".

### Power Systems

The swing equation `d delta/dt = omega` is literally a second-order Kuramoto model.
PMU phasor angles are oscillator phases; line admittances are coupling
constants.  No phase extraction step is needed -- measurement IS phase.
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
phases -- exactly the kind of synchronisation Kuramoto detects.

### Queue Networks

Service queues exhibit oscillatory behaviour under periodic demand.
Phase relationships between upstream and downstream queues determine
system throughput.  Spiked arrivals synchronise queue oscillations.

### Circadian Biology

SCN neurons are literal coupled oscillators with ~24 h period.  Clock
genes (Per/Cry, Bmal1, Rev-erb) form transcription-translation feedback
loops whose phase relationships determine circadian entrainment.
Winfree (1967); Strogatz (2003) *Sync* Ch. 5.

### Traffic Flow

Traffic signal coordination IS phase synchronisation.  Each signalised
intersection cycles with period ~90 s; offset-based green wave coordination
aligns phase differences between adjacent signals.  Gershenson &
Rosenblueth (2012) showed self-organising traffic lights converge via
coupled-oscillator dynamics.

### Epidemic SIR

Epidemic waves oscillate with well-defined periods driven by seasonal
forcing, immunity waning, and intervention cycles.  The SIR model produces
damped oscillations that are naturally phase-coupled across regions via
mobility.  Seasonal forcing acts as external drive (zeta).  Earn et al.
(2000).

### Biology (Bio Stub)

Biological systems oscillate at every scale: Ca2+ transients (ms),
cardiac rhythm (s), circadian clocks (24 h), hormonal cycles (days-weeks).
Multi-scale coupling between cellular, tissue, organ, and systemic layers
is inherently a Kuramoto hierarchy.

### Graph Geometry (Geometry Walk)

Random walkers on a graph synchronise when coupling exceeds a critical
threshold related to spectral gap.  Phase = ring-mapped node index
(theta = 2*pi*s/N).  Clustering and fragmentation transitions map to
Kuramoto order parameter bifurcations.

### Minimal Domain

Minimal 2-layer, 4-oscillator test harness exercising every pipeline
component: CouplingBuilder, UPDEEngine, BoundaryObserver, RegimeManager,
SupervisorPolicy, PolicyEngine.  Reference implementation for new
domain authors.

### Fusion Equilibrium

Grad-Shafranov equilibrium, MHD stability, transport, and ELM/sawtooth
events form a coupled oscillator hierarchy.  FusionCoreBridge maps
tokamak observables (q-profile, beta_N, tau_E) to oscillator phases.
ITER Physics Basis (2007).

### Laser Arrays

Semiconductor laser arrays couple via evanescent fields in shared
waveguide substrates.  Each laser's optical phase evolves under gain
competition and nearest-neighbour evanescent coupling — structurally
identical to Kuramoto with finite-range coupling.  Winful & Wang,
Appl Phys Lett 53(20), 1988; Kozyreff et al., PRL 85(18), 2000.

### PLL Clock Networks

Phase-locked loops track a reference clock by adjusting VCO frequency
proportional to phase error — exactly the Kuramoto coupling term
K·sin(θ_ref − θ_vco).  Hierarchical PLL networks (stratum clocks)
synchronise via cascaded phase detectors.  Strogatz & Mirollo, SIAM
J Appl Math 1988; ITU-T G.811.

### Firefly Swarms

Firefly flash synchronisation is the canonical biological Kuramoto
example.  Each firefly adjusts its flash-phase based on visual
coupling to neighbours, converging to collective synchrony.  Mirollo
& Strogatz (1990) proved global synchronisation for identical
pulse-coupled oscillators.

### Swarm Robotics

The Vicsek model — robots aligning heading angles with neighbours
plus noise — is a discrete-time Kuramoto model on a proximity graph.
Heading phase = oscillator phase; alignment = coupling.  Vicsek et al.,
PRL 75(6), 1995; Cucker & Smale, IEEE TAC 2007.

## Phase Extraction Rationale

| Domain | Source Signal | Extraction | Phase = |
|--------|-------------|-----------|---------|
| EEG | Voltage time series | Bandpass + Hilbert | Instantaneous phase |
| Power grid | PMU phasor | Direct measurement | Rotor angle delta |
| Cardiac | Intracardiac EGM | Activation time mapping | Activation phase |
| Rotating | Accelerometer | Order tracking + FFT | Harmonic phase |
| Chemical | T, C, P sensors | Detrend + Hilbert | Oscillation phase |
| Plasma | Mirnov coils, ECE | Mode fitting | Mode phase |
| Manufacturing | SPC sensor | Detrend around setpoint | Deviation phase |
| Queue | Queue depth | Detrend + Hilbert | Demand phase |
| Circadian | Clock gene expression | Cosinor fit | Acrophase |
| Traffic | Signal state | Cycle normalisation | Cycle phase |
| Epidemic | Case counts | Detrend + Hilbert | Wave phase |
| Biology | Multi-modal sensors | Scale-appropriate | Per-scale phase |
| Geometry | Graph node index | Ring mapping theta=2*pi*s/N | Node phase |
| Fusion | Diagnostic signals | Observable mapping | Equilibrium phase |
| Laser array | Optical field | Heterodyne interferometry | Optical phase |
| PLL clock | VCO output | Phase detector | VCO phase |
| Firefly | Flash events | Inter-flash interval | Flash phase |
| Swarm | IMU heading | Compass reading | Heading angle |

## Good/Bad Layer Partition

| Domain | Good (R up = healthy) | Bad (R up = pathological) |
|--------|--------------------|-----------------------|
| EEG | Alpha, gamma, network | Delta (wake), beta excess |
| Power grid | Generator sync, area freq | Load demand, renewable |
| Cardiac | SA node, atrial | Ventricular desync |
| Rotating | Shaft at nominal | Blade flutter, structural |
| Chemical | Heat transfer, feed flow | Kinetics oscillation |
| Plasma | Transport barrier, current | Turbulence, sawteeth |
| Manufacturing | Machine, line | Sensor (drift = bad) |
| Queue | Throughput (macro) | Retry burst (micro) |
| Circadian | SCN core, peripheral | Behavioral desync (jet lag) |
| Traffic | Corridor (green wave) | Intersection (gridlock), demand |
| Epidemic | Intervention coordination | Infection wave, mobility |
| Biology | Tissue, organ, systemic | (none defined) |
| Geometry | Local, global coherence | (none defined) |
| Minimal | Lower, upper | (none defined) |
| Fusion | Equilibrium, transport, boundary | Events (sawtooth, ELM) |
| Laser array | Single laser, array coupling | External cavity (feedback) |
| PLL clock | VCO lock, network PLL | Stratum hierarchy (holdover) |
| Firefly | Individual flash, swarm | (none defined) |
| Swarm | Heading alignment, flock direction | Formation breakup |

## Boundary Sources

| Domain | Hard Boundaries | Standard/Source |
|--------|----------------|----------------|
| EEG | Broadband sync < 0.9 | Lehnertz (2009) |
| Power grid | Freq +/-0.5 Hz, V 0.95-1.05 pu | NERC BAL-003-2, ANSI C84.1 |
| Cardiac | HR 40-180 bpm, QT < 500 ms | ACC/AHA guidelines, Roden (2004) |
| Rotating | Vibration < 7.1 mm/s | ISO 10816-3 zone C/D |
| Chemical | T < 450 C, P < 15 bar | Semenov limit, ASME VIII |
| Plasma | q_min >= 1, beta_N <= 2.8 | Kruskal-Shafranov, Troyon |
| Manufacturing | Temp < 85 C, pressure > 2 | OEM specs |
| Circadian | Phase deviation < 3 h | Clinical circadian disruption |
| Traffic | Queue < 50 vehicles | Intersection capacity |
| Epidemic | Cases < 100/100k, hospital < 80% | WHO threshold |
| Biology | HR 40-180 bpm | Clinical range |
| Fusion | q_min >= 1, beta_N <= 2.8 | Kruskal-Shafranov, Troyon |
| Laser array | Phase variance < 0.3 rad, feedback < 0.5 | Winful & Wang (1988) |
| PLL clock | Phase error < 100 ns, drift < 10 ppm | IEEE 1588, ITU-T G.811 |
| Firefly | Flash variance < 0.5 s | Observational ecology |
| Swarm | Formation error < 2 m, collision > 0.5 m | Safety standards |

## Actuator Mapping

| Domain | K (coupling) | alpha (lag) | zeta (drive) | Psi (target) |
|--------|-------------|------------|-------------|-------------|
| EEG | Connectivity | Delta band lag | Entrainment stim | Target phase |
| Power grid | Governor droop | Load shed phase | AGC bias | Curtailment |
| Cardiac | Drug coupling | Vagal modulation | Pacing rate | Pacing target |
| Rotating | Bearing stiffness | Damper viscosity | Speed setpoint | -- |
| Chemical | Coolant flow | Agitator speed | Feed rate | Jacket SP |
| Plasma | Global coupling | Turbulence lag | Damping | -- |
| Manufacturing | Global coupling | Sensor lag | Damping | -- |
| Queue | Global coupling | Micro lag | Damping | -- |
| Circadian | Inter-clock coupling | Sleep schedule lag | Light exposure | Meal timing |
| Traffic | Signal coordination | Phase split | Green wave offset | Ramp metering |
| Epidemic | Vaccination coordination | Travel restriction | Social measures | Lockdown target |
| Biology | Global coupling | -- | Entrainment | Reference phase |
| Geometry | Global coupling | -- | -- | -- |
| Minimal | Global coupling | -- | -- | -- |
| Fusion | Global coupling | -- | Entrainment | -- |
| Laser array | Evanescent coupling | Detuning offset | Injection current | Feedback phase |
| PLL clock | Loop bandwidth | Frequency trim | Reference drive | Phase target |
| Firefly | Visual coupling | -- | Environmental light | Flash target |
| Swarm | Alignment coupling | Obstacle avoidance | Formation drive | Target heading |

## Coupling Topology Rationale

| Domain | Topology | Rationale |
|--------|---------|-----------|
| EEG | Symmetric, non-negative | Cortical connectivity is undirected |
| Power grid | Distance-decayed | Admittance ~ 1/impedance ~ 1/distance |
| Cardiac | Strong nearest-neighbour | Gap junctions connect adjacent cells |
| Rotating | Layer-block | Mechanical path: shaft -> bearing -> structure |
| Chemical | Dense intra-layer | Heat-mass coupling is tight within reactions |
| Plasma | Hierarchical decay | Timescale separation between layers |
| Manufacturing | Weak cross-layer | Sensors are only indirectly coupled |
| Queue | Distance-decayed | Upstream/downstream proximity |
| Circadian | Strong intra-layer | Clock genes tightly coupled within SCN |
| Traffic | Distance-decayed | Adjacent intersections strongly coupled |
| Epidemic | Weak cross-layer | Regions coupled via mobility only |
| Biology | Hierarchical decay | Timescale separation across scales |
| Geometry | Distance-decayed | Graph adjacency determines coupling |
| Minimal | Distance-decayed | Default template |
| Fusion | Hierarchical decay | Timescale separation between layers |
| Laser array | Distance-decayed | Evanescent field exponential decay |
| PLL clock | Hierarchical decay | Stratum hierarchy (cascaded PLLs) |
| Firefly | Distance-decayed | Line-of-sight visual range |
| Swarm | Distance-decayed | Proximity-based communication range |

## Imprint Semantics

| Domain | Physical Meaning | Timescale | Modulates |
|--------|-----------------|-----------|-----------|
| EEG | Meditation training (plasticity) | Weeks-months | K, alpha |
| Cardiac | Drug accumulation (pharmacokinetics) | Hours-days | K |
| Chemical | Catalyst fouling/deactivation | Months | K |
| Manufacturing | Tool wear history | Weeks | K |
| Circadian | Chronic jet lag / shift work debt | Days-weeks | K, alpha |
| Biology | Chronic exposure accumulation | Days-months | K, alpha |
| Fusion | Plasma facing component erosion | Weeks | K |
| Laser array | Mirror degradation (facet erosion) | Months | K |
| PLL clock | Crystal aging (frequency drift) | Years | K, alpha |
| Firefly | (none — memoryless dynamics) | -- | -- |
| Swarm | (none — stateless dynamics) | -- | -- |
| Plasma | (none — fast relative to wall conditioning) | -- | -- |
| Power grid | Transformer insulation aging (IEEE C57.91) | Years | K |
| Rotating | Bearing wear (ISO 15243 spalling) | Weeks-months | K |
| Queue | Service degradation (memory leaks, pool exhaustion) | Hours-days | K |
| Traffic | Signal timing drift + road degradation | Weeks-months | K, alpha |
| Epidemic | Waning immunity (Antia et al. 2018) | Months | K |
| Geometry | (none) | -- | -- |
| Minimal | (none) | -- | -- |

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
