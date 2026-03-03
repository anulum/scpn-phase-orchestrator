# Quantum Simulation Domainpack

Maps qubit register phases to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Qubit XY-plane phases (Bloch sphere azimuthal angle) evolve under
Hamiltonian dynamics with coupling terms proportional to exchange
interaction strengths -- structurally identical to Kuramoto coupling.
Zhirov & Shepelyansky (PRL 2006) showed quantum synchronisation
transitions in coupled qubits mirror classical Kuramoto bifurcations.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| qubit_register | 4 | P (physical) | Raw qubit XY-plane phases |
| logical_coherence | 4 | I (informational) | Logical qubit entanglement events |

## Boundaries

- **fidelity_floor**: fidelity >= 0.5 (hard) -- minimum gate fidelity

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Exchange interaction strength |
| entrainment | zeta | External drive (microwave pulse) |

## Imprint

None. Qubit dynamics are unitary (memoryless on simulation timescale).

## Scenario

200 steps: coherent register -> decoherence onset -> fidelity drop ->
dynamical decoupling pulse -> coherence recovery.

## Optional Dependency

Install `scpn-quantum-control` for circuit construction and statevector
phase extraction via `QuantumControlBridge`.
