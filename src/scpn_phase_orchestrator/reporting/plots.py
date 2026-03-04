# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np

__all__ = ["CoherencePlot"]

_REGIME_COLORS = {
    "NOMINAL": "#2ecc71",
    "DEGRADED": "#f39c12",
    "CRITICAL": "#e74c3c",
    "RECOVERY": "#3498db",
}


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib required: pip install scpn-phase-orchestrator[plot]"
        )


class CoherencePlot:
    """Audit log visualisation from JSONL step records."""

    def __init__(self, log_data: list[dict]) -> None:
        self._data = log_data
        self._steps = [d for d in log_data if "step" in d and "layers" in d]

    def plot_r_timeline(self, output_path: str | Path) -> Path:
        """Line chart of per-layer R over simulation steps."""
        _require_matplotlib()
        steps = self._steps
        if not steps:
            raise ValueError("No step records in log data")

        x = [s["step"] for s in steps]
        n_layers = len(steps[0]["layers"])

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(n_layers):
            r_vals = [
                s["layers"][i]["R"] if i < len(s["layers"]) else 0.0 for s in steps
            ]
            ax.plot(x, r_vals, linewidth=0.8, label=f"L{i}")
        ax.set_xlabel("Step")
        ax.set_ylabel("R (order parameter)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=min(n_layers, 8), loc="upper right")
        fig.tight_layout()

        out = Path(output_path)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def plot_regime_timeline(self, output_path: str | Path) -> Path:
        """Coloured horizontal bands per regime epoch."""
        _require_matplotlib()
        steps = self._steps
        if not steps:
            raise ValueError("No step records in log data")

        fig, ax = plt.subplots(figsize=(10, 1.5))
        prev_regime = steps[0].get("regime", "NOMINAL")
        epoch_start = steps[0]["step"]

        def _flush(regime: str, start: int, end: int) -> None:
            color = _REGIME_COLORS.get(regime, "#95a5a6")
            ax.add_patch(Rectangle((start, 0), end - start, 1, color=color, alpha=0.7))

        for s in steps[1:]:
            regime = s.get("regime", "NOMINAL")
            if regime != prev_regime:
                _flush(prev_regime, epoch_start, s["step"])
                prev_regime = regime
                epoch_start = s["step"]
        _flush(prev_regime, epoch_start, steps[-1]["step"] + 1)

        ax.set_xlim(steps[0]["step"], steps[-1]["step"] + 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Step")

        # Legend
        for label, color in _REGIME_COLORS.items():
            ax.plot([], [], "s", color=color, label=label)
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
        fig.tight_layout()

        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_action_audit(self, output_path: str | Path) -> Path:
        """Vertical markers at steps where control actions fired."""
        _require_matplotlib()
        steps = self._steps
        if not steps:
            raise ValueError("No step records in log data")

        x_all = [s["step"] for s in steps]
        r_global = []
        for s in steps:
            rs = [la["R"] for la in s["layers"]]
            r_global.append(float(np.mean(rs)) if rs else 0.0)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_all, r_global, color="#2c3e50", linewidth=0.8, label="R_global")

        knob_colors: dict[str, str] = {}
        palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        for s in steps:
            for a in s.get("actions", []):
                knob = a["knob"]
                if knob not in knob_colors:
                    knob_colors[knob] = palette[len(knob_colors) % len(palette)]
                ax.axvline(s["step"], color=knob_colors[knob], alpha=0.4, linewidth=0.6)

        ax.set_xlabel("Step")
        ax.set_ylabel("R_global")
        ax.set_ylim(-0.05, 1.05)

        for knob, color in knob_colors.items():
            ax.plot([], [], color=color, linewidth=2, label=f"action: {knob}")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()

        out = Path(output_path)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def plot_amplitude_timeline(self, output_path: str | Path) -> Path:
        """Mean amplitude per step with subcritical threshold line.

        Reads 'mean_amplitude' from step records. Falls back to zero
        if the field is absent (phase-only simulation).
        """
        _require_matplotlib()
        steps = self._steps
        if not steps:
            raise ValueError("No step records in log data")

        x = [s["step"] for s in steps]
        amps = [s.get("mean_amplitude", 0.0) for s in steps]
        sub_frac = [s.get("subcritical_fraction", 0.0) for s in steps]

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(x, amps, color="#2980b9", linewidth=1.0, label="mean amplitude")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Mean amplitude", color="#2980b9")
        ax1.tick_params(axis="y", labelcolor="#2980b9")

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            sub_frac,
            color="#e74c3c",
            linewidth=0.8,
            linestyle="--",
            label="subcritical fraction",
        )
        ax2.set_ylabel("Subcritical fraction", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        fig.tight_layout()

        out = Path(output_path)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def plot_pac_heatmap(self, output_path: str | Path) -> Path:
        """N x N PAC modulation index heatmap.

        Reads the last 'pac_matrix' event/record from log data.
        The matrix should be stored as a flat list with shape (N, N).
        """
        _require_matplotlib()
        pac_record = None
        for d in reversed(self._data):
            if "pac_matrix" in d:
                pac_record = d
                break
        if pac_record is None:
            raise ValueError("No pac_matrix record in log data")

        flat = pac_record["pac_matrix"]
        n = pac_record.get("n", int(np.sqrt(len(flat))))
        mat = np.array(flat).reshape(n, n)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="viridis", aspect="equal", vmin=0.0)
        ax.set_xlabel("Amplitude oscillator")
        ax.set_ylabel("Phase oscillator")
        fig.colorbar(im, ax=ax, label="Modulation index")
        fig.tight_layout()

        out = Path(output_path)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
