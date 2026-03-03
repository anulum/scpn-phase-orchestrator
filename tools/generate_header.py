# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the synchronization manifold header image (1280x640 PNG).

Simulates Kuramoto oscillators transitioning from random phases to
phase-locked coherence — the mathematical ground truth of a Phase
Orchestrator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _REPO / "docs" / "assets" / "synchronization_manifold.png"
WIDTH, HEIGHT = 12.8, 6.4
DPI = 100


def generate(output: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="#08080c")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])

    # Background probability-density texture
    x = np.linspace(0, WIDTH, 200)
    y = np.linspace(0, HEIGHT, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-0.05 * ((X - WIDTH * 0.7) ** 2 + (Y - HEIGHT * 0.5) ** 2))
    ax.imshow(
        Z, extent=[0, WIDTH, 0, HEIGHT],
        origin="lower", cmap="magma", alpha=0.1, aspect="auto",
    )

    # Synchronized wavefronts: chaotic left → coherent right
    t = np.linspace(0, 4, 500)
    for i in range(8):
        offset_y = 1.5 + i * 0.5
        amplitude = 0.3 * (1 - (t / 4) * 0.5)
        phase_noise = rng.normal(0, 0.5 * (1 - (t / 4)))
        wave = offset_y + amplitude * np.sin(t * 8 + i * 0.2 + phase_noise)
        color = plt.cm.cool(i / 8.0)
        ax.plot(t + 7, wave, color=color, lw=1.5, alpha=0.7)

    # Chaotic input spikes
    for _ in range(25):
        sx = rng.uniform(1.0, 5.0)
        sy = rng.uniform(1.5, 5.0)
        ax.plot([sx, sx], [sy, sy + 0.3], color="white", alpha=0.2, lw=1)

    # Central orchestration hub — concentric phase rings
    hub = (6.0, 3.2)
    for r in [0.4, 0.8, 1.2]:
        ax.add_artist(plt.Circle(hub, r, color="#ffd700", fill=False, lw=1, alpha=0.3))
    for angle in np.linspace(0, 2 * np.pi, 12):
        ax.plot(
            [hub[0], hub[0] + 1.2 * np.cos(angle)],
            [hub[1], hub[1] + 1.2 * np.sin(angle)],
            color="#ffd700", alpha=0.2, lw=0.5,
        )

    # Title text
    ax.text(1.0, 5.4, "SCPN-PHASE-ORCHESTRATOR", color="#ffffff",
            fontsize=32, fontweight="bold", fontfamily="monospace", alpha=0.9)
    ax.text(1.0, 4.9, "TEMPORAL COHERENCE & PHASE-LOCKING ENGINE", color="#ffd700",
            fontsize=12, fontfamily="monospace", alpha=0.8)
    ax.text(1.0, 4.5, "// MODE: KURAMOTO_SYNC_ACTIVE", color="#00ffff",
            fontsize=10, fontfamily="monospace", alpha=0.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Generated {output} ({output.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synchronization manifold header image.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
