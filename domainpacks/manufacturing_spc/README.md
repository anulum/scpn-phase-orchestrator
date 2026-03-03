# Manufacturing SPC Domainpack

Statistical process control for discrete manufacturing lines, mapped to
SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Sensor signals (vibration, temperature, pressure) oscillate around
setpoints.  Tool wear causes systematic drift that correlates sensor
phases -- exactly the kind of synchronisation Kuramoto detects.
When sensors synchronise (high R on bad layer), it signals correlated
drift from a common root cause.

## Layers

| Layer | Oscillators | Channel | Source |
|-------|------------|---------|--------|
| sensor | 4 | P (physical) | Accelerometers, thermocouples, transducers |
| machine | 3 | I (informational) | PLC/SCADA: OEE, cycle time, throughput |
| line | 2 | S (symbolic) | Quality inspection: yield rate, defect class |

## Boundaries

- **temp_high**: temperature < 85°C (hard) -- OEM limit
- **pressure_low**: pressure > 2.0 bar (hard) -- minimum process pressure
- **vibration_warning**: vibration < 4.5 mm/s RMS (soft)

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-station signal coupling |
| lag_sensor | alpha | Sensor-to-controller delay |
| damping_global | zeta | PID damping injection |

## Imprint

Tool wear history: progressive degradation accumulates and modulates
coupling, representing gradual loss of machining precision.

## Scenario

200 steps: normal production -> systematic tool wear (sensor drift) ->
random machine failure -> golden sample reference injection -> recovery.
