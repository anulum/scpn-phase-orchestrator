#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Prometheus → QueueWaves Ingestion
#
# Shows how to ingest real Prometheus metrics into SPO's QueueWaves
# cascade failure detector. Uses synthetic metrics as a stand-in;
# replace with actual PromQL queries in production.
#
# Usage: python examples/prometheus_queuewaves.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def fetch_prometheus_metrics(
    n_services: int = 5,
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate fetching queue depth metrics from Prometheus.

    In production, replace with:
        import requests
        resp = requests.get(
            "http://prometheus:9090/api/v1/query_range",
            params={
                "query": "rate(http_requests_total[1m])",
                "start": start_ts, "end": end_ts, "step": "5s",
            },
        )
        data = resp.json()["data"]["result"]
    """
    rng = np.random.default_rng(seed)
    services = {}
    names = ["order-api", "payment-svc", "inventory", "shipping", "gateway"]
    for i, name in enumerate(names[:n_services]):
        base_rate = 50 + i * 10
        oscillation = 20 * np.sin(
            TWO_PI * (0.1 + rng.normal(0, 0.01)) * np.arange(n_samples)
        )
        noise = rng.standard_normal(n_samples) * 5
        services[name] = np.maximum(0, base_rate + oscillation + noise)
    return services


def queue_depth_to_phase(queue_depths: np.ndarray) -> float:
    """Convert queue depth time series to instantaneous phase.

    Uses the analytic signal (Hilbert transform) of the detrended
    queue depth. This is the Informational channel extraction.
    """
    from scipy.signal import hilbert

    detrended = queue_depths - np.mean(queue_depths)
    analytic = hilbert(detrended)
    return float(np.angle(analytic[-1]) % TWO_PI)


def main() -> None:
    print("Prometheus → QueueWaves Cascade Detector")
    print("=" * 50)

    n_services = 5
    metrics = fetch_prometheus_metrics(n_services=n_services)

    print(f"\nServices monitored: {list(metrics.keys())}")
    for name, depths in metrics.items():
        print(f"  {name}: mean={np.mean(depths):.1f} req/s")

    # Extract phases from queue depth oscillations
    phases = np.array([queue_depth_to_phase(depths) for depths in metrics.values()])

    # Build coupling: services that call each other couple strongly
    # order-api → payment, inventory; payment → shipping; gateway → all
    knm = np.zeros((n_services, n_services))
    knm[0, 1] = 2.0  # order → payment
    knm[0, 2] = 1.5  # order → inventory
    knm[1, 3] = 1.0  # payment → shipping
    knm[4, :] = 0.5  # gateway → all
    knm = 0.5 * (knm + knm.T)
    np.fill_diagonal(knm, 0.0)

    # Run SPO engine
    engine = UPDEEngine(n_services, dt=5.0)  # 5s scrape interval
    alpha = np.zeros((n_services, n_services))
    omegas = np.full(n_services, TWO_PI * 0.1)

    print("\nRunning cascade detection...")
    for epoch in range(5):
        # In production: fetch new metrics each tick
        new_metrics = fetch_prometheus_metrics(n_services=n_services, seed=epoch * 10)
        phases = np.array([queue_depth_to_phase(d) for d in new_metrics.values()])

        for _ in range(10):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        R, _ = compute_order_parameter(phases)
        risk = "CASCADE RISK" if R > 0.8 else "WARNING" if R > 0.6 else "OK"
        print(f"  Tick {epoch + 1}: R={R:.3f} [{risk}]")

    print("\nIn production, connect to real Prometheus via PromQL")
    print("and run as: spo queuewaves serve --config queuewaves.yaml")


if __name__ == "__main__":
    main()
