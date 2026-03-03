# Quantum Simulation Domainpack

Maps qubit register phases to SPO's Kuramoto/UPDE framework.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| qubit_register | q0–q3 | P | Raw qubit XY-plane phases |
| logical_coherence | log0–log3 | I | Logical qubit entanglement events |

## Boundary

- **fidelity_floor**: fidelity >= 0.5 (hard) — minimum gate fidelity

## Usage

```bash
spo validate domainpacks/quantum_simulation/binding_spec.yaml
python domainpacks/quantum_simulation/run.py
```

## Optional Dependency

Install `scpn-quantum-control` to enable circuit construction and
statevector phase extraction via `QuantumControlBridge`.
