# Domain Utilisation Schemas

Cross-domain comparison of all 33 SPO domainpacks, showing how Kuramoto/UPDE
phase dynamics map to diverse physical, biological, and engineered systems.

## Master Domainpack Table

| Pack | Layers | Osc | Safety Tier | Pipeline | Key Innovation |
|------|--------|-----|-------------|----------|---------------|
| agent_coordination | 3 | 12 | research | full | Multi-agent AI sync (heartbeat, task, topic) |
| autonomous_vehicles | 3 | 8 | research | full | Vehicle platoon phase-locking |
| bio_stub | 4 | 16 | research | full | Multi-scale biological oscillators |
| brain_connectome | 4 | 12 | research | full | HCP-inspired structural connectivity |
| cardiac_rhythm | 4 | 10 | clinical | full | Gap-junction coupling, arrhythmia |
| chemical_reactor | 4 | 10 | production | full | Hopf bifurcation, Semenov limit |
| circadian_biology | 4 | 10 | research | full | SCN clock-gene coupled oscillators |
| epidemic_sir | 3 | 8 | research | full | Epidemic wave synchronisation |
| financial_markets | 4 | 8 | research | full | Hilbert phase, crash regime detection |
| firefly_swarm | 2 | 8 | research | full | Mirollo-Strogatz flash synchronisation |
| fusion_equilibrium | 6 | 12 | research | full | MHD equilibrium + FusionCoreBridge |
| gene_oscillator | 3 | 6 | research | full | Repressilator + quorum sensing |
| geometry_walk | 2 | 8 | research | full | Random walk on graphs |
| identity_coherence | 6 | 35 | research | full | SSGF identity model, chimera + plasticity |
| laser_array | 3 | 8 | research | full | Evanescent-coupled laser phase-locking |
| manufacturing_spc | 3 | 9 | consumer | full | SPC process drift detection |
| metaphysics_demo | 3 | 7 | research | full | P/I/S + imprint + geometry |
| minimal_domain | 2 | 4 | research | full | Minimal-but-complete pipeline example |
| musical_acoustics | 3 | 9 | research | full | Consonance and groove via harmonic sync |
| network_security | 3 | 8 | research | full | DDoS detection via traffic sync anomaly |
| neuroscience_eeg | 6 | 14 | research | full | EEG band->phase, seizure detection |
| plasma_control | 8 | 16 | research | adapter | Full tokamak layer hierarchy (PlasmaControlBridge) |
| pll_clock | 3 | 8 | production | full | PLL network clock sync (ITU-T G.811) |
| power_grid | 5 | 12 | production | full | Swing equation = Kuramoto (exact) |
| quantum_simulation | 3 | 8 | research | adapter | Quantum gate phase tracking (QuantumControlBridge) |
| queuewaves | 3 | 6 | consumer | full | Service queue oscillations |
| robotic_cpg | 4 | 8 | consumer | full | Quadruped CPG locomotion gait patterns |
| rotating_machinery | 4 | 10 | consumer | full | Vibration harmonics, ISO 10816 |
| satellite_constellation | 3 | 8 | research | full | Orbital slot + comms link sync |
| sleep_architecture | 4 | 8 | research | full | AASM sleep staging from R values |
| swarm_robotics | 3 | 8 | consumer | full | Vicsek collective motion/formation |
| traffic_flow | 4 | 10 | consumer | full | Signal coordination = phase sync |
| vortex_shedding | 3 | 9 | research | full | Wake dynamics (Stuart-Landau amplitude) |

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

### Autonomous Vehicles

Vehicle platoons maintain fixed headway via adaptive cruise control.
Each vehicle's position oscillates around the desired following distance
— a phase-coupled system where synchronisation = stable platoon and
desynchronisation = collision risk or string instability.  Dey et al.,
IEEE Trans Intell Transp Syst 17(5), 2016.

### Brain Connectome

HCP-derived structural connectivity matrices define coupling topology
between cortical regions.  Each region oscillates at characteristic
frequencies; the connectome determines which regions phase-lock.
Bullmore & Sporns, Nature Reviews Neuroscience 10, 2009.

### Financial Markets

Asset returns exhibit collective synchronisation preceding crashes.
Hilbert-transformed price series yield instantaneous phases; the
Kuramoto order parameter R(t) → 1 signals herding behaviour.
Harmon et al., PLoS ONE 6(4), 2011.

### Gene Oscillator

The repressilator (Elowitz & Leibler, Nature 403, 2000) is a synthetic
three-gene oscillatory circuit.  Quorum sensing couples repressilators
across cells — structurally identical to Kuramoto coupling with
chemical diffusion as the coupling channel.

### Identity Coherence

The SSGF identity model treats cognitive traits (working style,
reasoning patterns, values) as oscillators whose synchronisation
defines coherent identity.  Chimera states (partial coherence) model
cognitive dissonance.  An application of the SCPN consciousness
framework to AI self-modelling.

### Musical Acoustics

Harmonic modes of musical instruments (fundamental, overtones) form
coupled oscillators.  Consonance = integer frequency ratios = specific
phase relationships.  Rhythmic groove emerges from synchronised beat
subdivisions.  Large & Palmer, Ecological Psychology 14(1-2), 2002.

### Network Security

Normal network traffic oscillates with diurnal and weekly periods.
DDoS attacks disrupt these patterns — anomalous synchronisation in
packet arrivals signals coordinated attack traffic.  Phase-based
detection complements rate-based methods.

### Robotic CPG

Central Pattern Generators produce rhythmic locomotion via coupled
oscillators.  Quadruped gaits (walk, trot, gallop) correspond to
specific phase relationships between leg CPGs — exactly Kuramoto with
discrete symmetry-breaking modes.  Ijspeert, Neural Networks 21(4),
2008.

### Satellite Constellation

Satellites in constellation maintain orbital slot phasing and
inter-satellite link timing.  Orbital mechanics produces oscillatory
relative motion; communication link synchronisation requires phase
coordination across the constellation.

### Sleep Architecture

EEG power in delta, theta, alpha, and beta bands defines AASM sleep
stages.  The order parameter R computed per band tracks transitions
between wake, N1, N2, N3, and REM — a natural Kuramoto hierarchy with
sleep stage = regime.

### Vortex Shedding

Karman vortex streets behind bluff bodies produce periodic lift and
drag oscillations at the Strouhal frequency.  Multiple cylinders
interact via wake coupling — Stuart-Landau amplitude dynamics capture
lock-in and vortex-induced vibration.  Williamson & Govardhan, Annual
Review of Fluid Mechanics 36, 2004.

### Agent Coordination

Multiple AI agents (Claude, Codex, Gemini, human) working on a shared
codebase exhibit oscillatory task patterns — heartbeat liveness checks,
task flow cycles, topic focus shifts.  Synchronisation = coordinated
parallel work; desynchronisation = merge conflicts and wasted effort.

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
| Autonomous vehicles | Headway distance | Detrend + Hilbert | Following phase |
| Brain connectome | fMRI BOLD / EEG | Bandpass + Hilbert | Regional phase |
| Financial markets | Price returns | Hilbert transform | Asset phase |
| Gene oscillator | Fluorescence reporter | Peak detection | Expression phase |
| Identity coherence | Trait activation signals | Inter-event frequency | Trait phase |
| Musical acoustics | Audio waveform | FFT harmonic tracking | Harmonic phase |
| Network security | Packet timestamps | Inter-arrival frequency | Traffic phase |
| Robotic CPG | Joint angle encoders | Direct measurement | Joint phase |
| Satellite constellation | Orbital position | Kepler elements | Orbital phase |
| Sleep architecture | EEG band power | Bandpass + Hilbert | Band phase |
| Vortex shedding | Pressure/force transducer | Detrend + Hilbert | Shedding phase |
| Agent coordination | Heartbeat timestamps | Inter-event frequency | Agent liveness phase |

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
| Circadian | SCN core, peripheral | Behavioural desync (jet lag) |
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
| Autonomous vehicles | Leader-follower platoon | String instability |
| Brain connectome | Visual, auditory, DMN sync | Hypersync (seizure) |
| Financial markets | Sector diversification | Cross-asset herding (crash) |
| Gene oscillator | Repressilator rhythm | Quorum desync |
| Identity coherence | Working style, values | Cognitive dissonance (chimera) |
| Musical acoustics | Harmonic consonance, groove | Dissonance, tempo drift |
| Network security | Normal traffic rhythm | Attack synchronisation |
| Robotic CPG | Gait phase coordination | Leg collision, stumble |
| Satellite constellation | Orbital slot, comms link | Constellation breakup |
| Sleep architecture | Delta (N3), alpha (wake) | Beta excess (insomnia) |
| Vortex shedding | Upstream wake coherence | Lock-in (structural fatigue) |
| Agent coordination | Task flow, topic alignment | Merge conflicts, duplicated work |

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
| Autonomous vehicles | Headway > 1.5 s, speed delta < 10 km/h | SAE J3016, ISO 22839 |
| Brain connectome | Global sync < 0.9 | Lehnertz (2009) |
| Financial markets | Drawdown < 5%, VIX < 30 | Risk management |
| Gene oscillator | Expression ratio 0.1-10x | Elowitz & Leibler (2000) |
| Identity coherence | R_identity > 0.3 | SSGF threshold |
| Musical acoustics | Intonation < 20 cents, tempo drift < 5% | Perceptual thresholds |
| Network security | Packet rate < 10x baseline | IDS thresholds |
| Robotic CPG | Joint angle limits, torque < max | Actuator specs |
| Satellite constellation | Slot drift < 0.1 deg, link margin > 3 dB | ITU Radio Regulations |
| Sleep architecture | Stage duration within AASM norms | AASM manual v3 |
| Vortex shedding | Amplitude < fatigue limit, St = 0.2 +/- 0.05 | ASME PTC 19.3 |
| Agent coordination | Heartbeat interval < 60 s, conflict rate < 0.1 | Operational SLA |

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
| Autonomous vehicles | Platoon coupling | Headway lag | ACC setpoint | Leader speed |
| Brain connectome | Connectivity strength | Propagation delay | Stimulation | Target region |
| Financial markets | Portfolio correlation | Sector rotation lag | Hedging | Risk target |
| Gene oscillator | Quorum coupling | Diffusion delay | Inducer concentration | -- |
| Identity coherence | Trait coupling | Cognitive lag | External feedback | Value target |
| Musical acoustics | Ensemble coupling | Tempo offset | Conductor beat | Pitch target |
| Network security | Traffic coupling | Routing lag | Rate limiting | Baseline pattern |
| Robotic CPG | Inter-leg coupling | Gait phase offset | Speed command | Gait target |
| Satellite constellation | Orbital coupling | Propagation delay | Thrust manoeuvre | Slot target |
| Sleep architecture | Inter-band coupling | Band transition lag | Light/sound stim | Sleep stage |
| Vortex shedding | Wake coupling | Convective delay | Flow speed | -- |
| Agent coordination | Task coupling | Communication lag | Priority signal | Coordination target |

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
| Autonomous vehicles | Distance-decayed | V2V range limited by proximity |
| Brain connectome | HCP-weighted | Structural connectivity from diffusion MRI |
| Financial markets | Correlation-based | Asset return correlation matrix |
| Gene oscillator | Dense intra-layer | Same-cell gene products diffuse freely |
| Identity coherence | Hierarchical decay | Trait layers separated by abstraction level |
| Musical acoustics | Strong intra-layer | Harmonics of same instrument tightly coupled |
| Network security | Distance-decayed | Network topology determines traffic paths |
| Robotic CPG | Strong nearest-neighbour | Adjacent legs mechanically coupled |
| Satellite constellation | Distance-decayed | Inter-satellite link range |
| Sleep architecture | Hierarchical decay | Timescale separation between EEG bands |
| Vortex shedding | Distance-decayed | Wake interaction decays with cylinder spacing |
| Agent coordination | Weak cross-layer | Agents coupled only via shared repo |

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
| Autonomous vehicles | Tyre wear, brake fade | Hours-days | K |
| Brain connectome | Synaptic plasticity (LTP/LTD) | Days-weeks | K, alpha |
| Financial markets | Regime memory (bull/bear momentum) | Weeks-months | K |
| Gene oscillator | Epigenetic modification | Days-weeks | K |
| Identity coherence | Trait reinforcement / habit formation | Weeks-months | K, alpha |
| Musical acoustics | Ensemble rehearsal (muscle memory) | Days-weeks | K |
| Network security | Baseline drift (traffic pattern evolution) | Days | K |
| Robotic CPG | Joint wear, actuator fatigue | Weeks-months | K |
| Satellite constellation | Orbital decay, component aging | Months-years | K |
| Sleep architecture | Chronic sleep debt | Days-weeks | K, alpha |
| Vortex shedding | Structural fatigue accumulation | Months | K |
| Agent coordination | (none — stateless coordination) | -- | -- |

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
