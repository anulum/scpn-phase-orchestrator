# GSE2565 phosgene lung — real bulk-DNB early-warning evidence

This directory holds the **bulk-transcriptomic** companion to the single-cell Mojtahedi
proof (`../mojtahedi_fate/`), and the case the dynamical network biomarker was first
demonstrated on: Chen, Liu, Liu & Aihara 2012 read a rising DNB composite index on
GSE2565, a phosgene acute-lung-injury time course, peaking at the ~8 h critical
transition. Here that benchmark is run through the programme's honest lens, and the point
of this directory is the **null it is tested against**.

## Why the usual reading is not enough

A bulk DNB analysis has **selection freedom**: the composite index is evaluated on a
*module* of genes, and the module is *chosen to peak at the transition*. So an apparent
rise-to-the-transition is partly guaranteed by construction — given enough genes and a
target timepoint, some module will always look like it rises. The only honest test
re-runs the **entire module selection on every surrogate**. This capstone does exactly
that: the null shuffles the timepoint labels across an arm's samples and re-selects the
module from scratch on the shuffled data, so a surrogate enjoys the same selection freedom
as the real analysis.

## The result, stated honestly

On the rising limb up to the ~8 h transition (0.5, 1, 4, 8 h), the DNB module is selected
at 8 h on the arm's 150 most-variable probes, its composite-index trajectory is reduced to
a least-squares slope, and that slope is ranked against 1 000 selection-controlled
surrogates:

| Arm | Observed rising slope | Surrogate 90th pct | Surrogate-rank p | Alarmed at 10 % FA |
|-----|----------------------:|-------------------:|-----------------:|:------------------:|
| CG (phosgene exposed) | 1.154 | 1.776 | **0.385** | — |
| Air (control) | 0.966 | 1.799 | 0.414 | — |

The phosgene-exposed arm's DNB rise **does not beat** the selection-controlled surrogates
(p = 0.385), and it barely exceeds the air control's (0.966): both observed slopes sit
below their own surrogate 90th percentile. Once the null is allowed the same freedom to
cherry-pick a rising module on scrambled time, the celebrated bulk DNB rise is largely a
**selection artefact**. This is the same finding as the four physical domains and the
single-cell arm, in a bulk-transcriptomic modality — a large-looking early-warning signal
is not an operational alarm that beats a properly controlled chance model.

An honest limitation: there is **one exposed arm**, so this is a single-transition
surrogate test, not a corpus — it bounds demonstrated skill on this record, not a
population claim. Its value is the selection-controlled null, which the "the DNB index
rose" reading omits.

## Reproduce

```bash
PYTHONPATH=src:. python -m bench.early_warning_dnb_bulk <dir-with-GSE2565-files> \
    examples/real_data/gse2565_lung
```

where `<dir-with-GSE2565-files>` holds the citation-only `GSE2565_series_matrix.txt.gz`
and `GSE2565_family.soft.gz` from GEO. The pipeline is deterministic (seed 0), so the
sealed `content_hash`
(`e5180ff4a7896acceba7da628207705129c8a22744a6e57e604898c623970b1f`) is reproduced; the
integrity test `tests/test_gse2565_dnb_evidence.py` recomputes and pins it. No raw data is
committed.

## References

* Chen, Liu, Liu & Aihara 2012, *Sci Rep* 2:342 — the dynamical network biomarker,
  demonstrated on GSE2565.
* Sciuto et al. 2005, *Chem Res Toxicol* — the GSE2565 phosgene lung-injury time course.
