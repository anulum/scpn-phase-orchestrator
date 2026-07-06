# Mojtahedi 2016 single-cell fate — real DNB early-warning evidence

This directory holds the **fifth domain** the SCPN Phase Orchestrator's early-warning
design was proven on, and the **first molecular** one. The scalp-EEG, cardiac-ECG,
grid-PMU and palaeoclimate capstones read *many time-samples* of a physical signal; a
cell-fate transition is the opposite regime — *few timepoints, many genes* — so the
sliding-window critical-slowing-down monitor does not apply. The **dynamical network
biomarker** (DNB) does: as EML progenitors approach an erythroid/myeloid fate
bifurcation, the single-cell transition index (mean gene–gene over cell–cell
correlation) rises sharply to a peak at the tipping point. This capstone runs that
celebrated index through the **identical matched-false-alarm and label-permutation
protocol** the four physical domains are scored by, and seals the result into a
hash-addressed record.

## What the input is (and is not)

The transition-index trajectory is taken **from the published supplement** — Mojtahedi et
al. 2016, *PLoS Biol* 14:e2000640, Table S2: the per-lineage transition index and its
bootstrap standard error at Days 0, 1, 3, 6 — **not re-derived from the raw single-cell
qPCR**. That is a deliberate, honest choice: the raw Ct matrix carries heavy non-detects
and an unspecified normalisation, so a re-derivation reproduces the published rise only
qualitatively; the authors' peer-reviewed index values are the sounder input. The index
*definition* is reproduced and validated in code — `bench.dnb_detector.
single_cell_transition_index` computes exactly the published form (verified: the published
transition index equals the ratio of the two published mean correlations). **No raw data
is redistributed here**; only the derived, hash-sealed artefact is committed.

## The result, stated honestly

The rising limb (Days 0, 1, 3, up to the peak) of each lineage is the operational
pre-transition window; its least-squares slope is the score; the matched false-alarm
threshold (≤ 10 %) is calibrated on 6 000 temporally-shuffled surrogates drawn from the
published means and standard errors:

| Lineage | Rising-limb slope | Alarmed at matched FA |
|---------|------------------:|:---------------------:|
| erythroid_epo | 0.128 | ✓ |
| myeloid_gmcsf_il3 | 0.092 | — |
| combined_epo_gmcsf | 0.028 | — |

**One of the three lineages** (the erythroid EPO arm, the strongest riser) clears the
10 % operating point; a label-permutation test over the three real transitions gives
**p = 0.266** — not significant. The single-cell transition index rises 2.8-fold toward
the bifurcation, an unmistakable effect size, yet at its published four-timepoint
resolution the rise does **not** reach corpus-level significance at a matched false-alarm
rate. This is the same finding as the four physical domains, in a fifth modality: a large,
real early-warning signal is not the same as an operational alarm that beats chance at a
controlled false-alarm budget. The binding constraint here is the **temporal
resolution**, not the effect size — three lineages and four timepoints give the test very
low power by construction, so this silence bounds demonstrated skill; it is not a proof of
impossibility.

## The grid modal-growth moat cannot be posed here

`dnb_modal_transfer.json` seals a second, negative result: a test of whether the
**power-grid** detector's winning form — the exponential envelope-growth rate and the
fit-quality (R²) gate that certifies its streaming operating point — transfers to this DNB
signal. It scores the transition-index rising limb by the exponential growth rate of its
envelope (the grid statistic) instead of a linear slope, over both DNB corpora: the
three-point single-cell limbs above and the four-point bulk GSE2565 exposed-arm limb
(`bench.early_warning_dnb_bulk`).

It cannot even be posed. On three-to-four points the fit-quality gate is **uninformative**:

| Trajectory | Points | Exp-fit R² | Gate keeps? |
|------------|-------:|-----------:|:-----------:|
| erythroid_epo | 3 | 0.82 | ✓ |
| myeloid_gmcsf_il3 | 3 | 0.86 | ✓ |
| combined_epo_gmcsf | 3 | 0.67 | ✓ |
| gse2565_cg_exposed | 4 | 0.80 | ✓ |

Every monotone rise fits an exponential well enough to pass a 0.5 gate, so the gate keeps
all four and rejects none — the discrimination it is meant to provide is gone. The
exponential growth rate, moreover, re-orders the three single-cell lineages in the same
order as the linear slope, adding no separating information: the transfer collapses to the
slope the existing detector already uses. The DNB rise is a genuine critical-slowing-down
divergence — a power law, not an exponential — and its early-warning trajectories are too
short to fit or gate a growth form. Together with the scalp-EEG result (a resolved but
non-exponential trajectory the gate rejects), this bounds the grid moat: it needs both a
genuine exponential instability and a resolved trajectory, which the power grid has and
these domains do not. Regenerate it (deterministic):

```bash
python bench/dnb_modal_transfer.py DATA examples/real_data/mojtahedi_fate/dnb_modal_transfer.json
```

with `DATA` holding the GSE2565 files; the single-cell limbs are read from the in-code
published Table S2, so no raw single-cell data is needed.

## Reproduce

```bash
PYTHONPATH=src:. python -m bench.early_warning_dnb examples/real_data/mojtahedi_fate
```

The capstone reads only the embedded published summary, so it runs with no external data
and is byte-reproducible: the sealed `content_hash`
(`353b2e7c6c62252b168047e6a7eb4d8a9881c8dc3e4d85477f3694302fab3f26`) is recomputed by the
integrity test `tests/test_mojtahedi_dnb_evidence.py`, which fails if the artefact is
hand-edited or the pipeline drifts.

## References

* Mojtahedi, Skupin, Zhou, Castaño, Leong-Quong, Chang, … Huang 2016, *PLoS Biol*
  14:e2000640 — the single-cell transition index at a leukaemic fate bifurcation.
* Chen, Liu, Liu & Aihara 2012, *Sci Rep* 2:342 — the dynamical network biomarker.
