// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hodge simplicial input admission

//! Cardinality, finite measurements and oriented simplicial support for Hodge flows.

use std::collections::HashSet;

/// Admit finite flattened measurements and unique canonical simplices.
///
/// # Errors
///
/// Refuses inconsistent dimensions, nonfinite values, non-ascending or duplicate
/// simplices and triangles unsupported by the supplied edge complex.
pub(crate) fn validate_hodge_inputs(
    knm_flat: &[f64],
    phases: &[f64],
    n: usize,
    edges: &[i64],
    n_edges: usize,
    tris: &[i64],
    n_tris: usize,
) -> Result<(), String> {
    let coupling_size = n.checked_mul(n).ok_or("n*n overflows usize")?;
    let edge_size = n_edges.checked_mul(2).ok_or("edge count overflows usize")?;
    let triangle_size = n_tris
        .checked_mul(3)
        .ok_or("triangle count overflows usize")?;
    if knm_flat.len() != coupling_size
        || phases.len() != n
        || edges.len() != edge_size
        || tris.len() != triangle_size
    {
        return Err("Hodge input cardinality does not match declared counts".into());
    }
    if knm_flat
        .iter()
        .chain(phases)
        .any(|value| !value.is_finite())
    {
        return Err("Hodge measurements must contain only finite values".into());
    }
    let mut edge_set = HashSet::with_capacity(n_edges);
    for &[i, j] in edges.as_chunks::<2>().0 {
        if i < 0 || i >= j || j as u64 >= n as u64 {
            return Err("Hodge edges must satisfy 0 <= i < j < n".into());
        }
        if !edge_set.insert((i, j)) {
            return Err("Hodge edges must be unique".into());
        }
    }
    let mut triangle_set = HashSet::with_capacity(n_tris);
    for &[i, j, k] in tris.as_chunks::<3>().0 {
        if i < 0 || i >= j || j >= k || k as u64 >= n as u64 {
            return Err("Hodge triangles must satisfy 0 <= i < j < k < n".into());
        }
        if !triangle_set.insert((i, j, k)) {
            return Err("Hodge triangles must be unique".into());
        }
        if ![(i, j), (i, k), (j, k)]
            .iter()
            .all(|edge| edge_set.contains(edge))
        {
            return Err("Hodge triangle references a missing edge".into());
        }
    }
    Ok(())
}
