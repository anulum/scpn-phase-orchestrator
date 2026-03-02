# Manufacturing SPC Domainpack

Statistical Process Control for discrete manufacturing lines.

## Layers

| Layer | Channel | Oscillators | Source |
|-------|---------|-------------|--------|
| sensor | P | vibration, temperature, pressure, flow_rate | Accelerometers, thermocouples, transducers |
| machine | I | oee, cycle_time, throughput | PLC/SCADA event streams |
| line | S | yield_rate, defect_class | Quality inspection states |

## Boundaries

- **temp_high** (hard): temperature < 85 C
- **pressure_low** (hard): pressure > 2.0 bar
- **vibration_warning** (soft): vibration < 4.5 mm/s RMS

## Usage

```bash
spo validate domainpacks/manufacturing_spc/binding_spec.yaml
spo run domainpacks/manufacturing_spc/binding_spec.yaml --steps 1000
```
