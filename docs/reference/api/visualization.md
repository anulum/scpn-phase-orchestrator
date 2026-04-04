# Visualization

The visualization subsystem provides real-time tools for monitoring the
topological state of oscillator networks.

## WebXR Manifold Streamer

The `VisualizerStreamer` module implements **Phase 11: Holographic
Projection** of the SCPN roadmap. It provides a high-frequency WebSocket
interface for streaming simulation telemetry to 3D front-ends (WebGL/Three.js).

### Features
- **60Hz Telemetry:** Broadcasts phase states and metric tensors at
  interactive frame rates.
- **Topological torus mapping:** Enables the projection of N-dimensional
  phase manifolds into human-readable 3D structures.
- **Metric Tensor Visualization:** Streams the rank-2 metric tensor $h_{\mu\nu}$
  to visualize local curvature and folding of the synchronization manifold.

### Usage Example

```python
from scpn_phase_orchestrator.visualization import VisualizerStreamer

# Initialize and start streamer
streamer = VisualizerStreamer(port=8765)
streamer.start()

# In the simulation loop:
while True:
    state = engine.step(...)
    metrics = observer.observe(...)

    # Broadcast to all connected WebXR clients
    streamer.broadcast({
        "phases": state,
        "curvature": metrics.gauge_curvature,
        "h_munu": metrics.h_munu
    })
```

::: scpn_phase_orchestrator.visualization.streamer

## Network Graph Visualization

D3-based network graph visualization for small to medium topologies.

::: scpn_phase_orchestrator.visualization.network

## Torus Visualization

Three.js-based 3D torus visualization for phase-space embedding.

::: scpn_phase_orchestrator.visualization.torus
