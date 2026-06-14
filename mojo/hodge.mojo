# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Combinatorial Hodge decomposition (Mojo port)

"""Combinatorial (Helmholtz–Hodge) decomposition of the Kuramoto
coupling current as a Mojo executable, backed by LAPACK ``dsyev_`` for
the symmetric positive semidefinite pseudoinverse.

Stdin:

    HODGE n n_edges n_tris knm[0..n*n] phases[0..n]
        edges[0..2*n_edges] tris[0..3*n_tris]

Prints ``3 * n * n`` f64 values: the flattened row-major gradient,
curl, and harmonic flow matrices in order. Node indices in ``edges``
and ``tris`` are 0-based.

Build with::

    mojo build mojo/hodge.mojo -o mojo/hodge_mojo
"""

from std.ffi import OwnedDLHandle
from std.math import sin
from std.collections import List

alias PINV_RCOND = 1e-9


fn psd_pinv_apply(mat: List[Float64], dim: Int, vec: List[Float64]) raises -> List[Float64]:
    """Apply pinv of a symmetric PSD row-major ``dim×dim`` matrix to
    ``vec`` via LAPACK ``dsyev_``."""
    var out = List[Float64](capacity=dim)
    for _ in range(dim):
        out.append(0.0)
    if dim == 0:
        return out^

    var a = List[Float64](capacity=dim * dim)
    for i in range(dim * dim):
        a.append(mat[i])

    var lib = OwnedDLHandle("/usr/lib/x86_64-linux-gnu/liblapack.so.3")
    var jobz = List[Int8](capacity=1); jobz.append(Int8(ord("V")))
    var uplo = List[Int8](capacity=1); uplo.append(Int8(ord("U")))
    var n_c = List[Int32](capacity=1); n_c.append(Int32(dim))
    var lda = List[Int32](capacity=1); lda.append(Int32(dim))
    var lwork_len = 3 * dim * dim + 64
    var lwork = List[Int32](capacity=1); lwork.append(Int32(lwork_len))
    var info = List[Int32](capacity=1); info.append(0)
    var w = List[Float64](capacity=dim)
    for _ in range(dim):
        w.append(0.0)
    var work = List[Float64](capacity=lwork_len)
    for _ in range(lwork_len):
        work.append(0.0)

    _ = lib.call["dsyev_", NoneType](
        jobz.unsafe_ptr(),
        uplo.unsafe_ptr(),
        n_c.unsafe_ptr(),
        a.unsafe_ptr(),
        lda.unsafe_ptr(),
        w.unsafe_ptr(),
        work.unsafe_ptr(),
        lwork.unsafe_ptr(),
        info.unsafe_ptr(),
    )
    if Int(info[0]) != 0:
        raise Error("dsyev failed")

    var lambda_max = w[dim - 1]
    var cutoff = 0.0
    if lambda_max > 0.0:
        cutoff = PINV_RCOND * lambda_max

    # Eigenvector k component r is a[k*dim + r] (column-major).
    var projected = List[Float64](capacity=dim)
    for k in range(dim):
        var p = 0.0
        if w[k] > cutoff:
            var acc = 0.0
            for r in range(dim):
                acc += a[k * dim + r] * vec[r]
            p = acc / w[k]
        projected.append(p)
    for r in range(dim):
        var acc = 0.0
        for k in range(dim):
            acc += a[k * dim + r] * projected[k]
        out[r] = acc
    return out^


fn find_edge(edge_i: List[Int], edge_j: List[Int], n_edges: Int, a: Int, b: Int) -> Int:
    for e in range(n_edges):
        if edge_i[e] == a and edge_j[e] == b:
            return e
    return -1


fn hodge_decomposition(
    knm: List[Float64],
    phases: List[Float64],
    n: Int,
    edges: List[Int],
    n_edges: Int,
    tris: List[Int],
    n_tris: Int,
    mut grad_m: List[Float64],
    mut curl_m: List[Float64],
    mut harm_m: List[Float64],
) raises -> None:
    var edge_i = List[Int](capacity=n_edges)
    var edge_j = List[Int](capacity=n_edges)
    var flow = List[Float64](capacity=n_edges)
    for e in range(n_edges):
        var i = edges[2 * e]
        var j = edges[2 * e + 1]
        edge_i.append(i)
        edge_j.append(j)
        var ksym = 0.5 * (knm[i * n + j] + knm[j * n + i])
        flow.append(ksym * sin(phases[j] - phases[i]))

    var l0 = List[Float64](capacity=n * n)
    for _ in range(n * n):
        l0.append(0.0)
    var divf = List[Float64](capacity=n)
    for _ in range(n):
        divf.append(0.0)
    for e in range(n_edges):
        var i = edge_i[e]
        var j = edge_j[e]
        l0[i * n + i] += 1.0
        l0[j * n + j] += 1.0
        l0[i * n + j] -= 1.0
        l0[j * n + i] -= 1.0
        divf[i] -= flow[e]
        divf[j] += flow[e]
    var potential = psd_pinv_apply(l0, n, divf)
    var f_grad = List[Float64](capacity=n_edges)
    for e in range(n_edges):
        f_grad.append(potential[edge_j[e]] - potential[edge_i[e]])

    var f_curl = List[Float64](capacity=n_edges)
    for _ in range(n_edges):
        f_curl.append(0.0)
    if n_tris > 0:
        # Signed edge entries of B2 per triangle, flattened (3 per triangle).
        var tri_edge = List[Int](capacity=3 * n_tris)
        var tri_sign = List[Float64](capacity=3 * n_tris)
        for t in range(n_tris):
            var i = tris[3 * t]
            var j = tris[3 * t + 1]
            var k = tris[3 * t + 2]
            tri_edge.append(find_edge(edge_i, edge_j, n_edges, i, j)); tri_sign.append(1.0)
            tri_edge.append(find_edge(edge_i, edge_j, n_edges, j, k)); tri_sign.append(1.0)
            tri_edge.append(find_edge(edge_i, edge_j, n_edges, i, k)); tri_sign.append(-1.0)
        var l2 = List[Float64](capacity=n_tris * n_tris)
        for _ in range(n_tris * n_tris):
            l2.append(0.0)
        var c2 = List[Float64](capacity=n_tris)
        for _ in range(n_tris):
            c2.append(0.0)
        for t in range(n_tris):
            for a in range(3):
                c2[t] += tri_sign[3 * t + a] * flow[tri_edge[3 * t + a]]
            for u in range(n_tris):
                var acc = 0.0
                for a in range(3):
                    for b in range(3):
                        if tri_edge[3 * t + a] == tri_edge[3 * u + b]:
                            acc += tri_sign[3 * t + a] * tri_sign[3 * u + b]
                l2[t * n_tris + u] = acc
        var tri_pot = psd_pinv_apply(l2, n_tris, c2)
        for t in range(n_tris):
            for a in range(3):
                f_curl[tri_edge[3 * t + a]] += tri_sign[3 * t + a] * tri_pot[t]

    for e in range(n_edges):
        var i = edge_i[e]
        var j = edge_j[e]
        var g = f_grad[e]
        var c = f_curl[e]
        var h = flow[e] - g - c
        grad_m[i * n + j] = g
        grad_m[j * n + i] = -g
        curl_m[i * n + j] = c
        curl_m[j * n + i] = -c
        harm_m[i * n + j] = h
        harm_m[j * n + i] = -h


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "HODGE":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var n_edges = Int(atol(tokens[idx])); idx += 1
    var n_tris = Int(atol(tokens[idx])); idx += 1

    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var edges = List[Int](capacity=2 * n_edges)
    for _ in range(2 * n_edges):
        edges.append(Int(atol(tokens[idx]))); idx += 1
    var tris = List[Int](capacity=3 * n_tris)
    for _ in range(3 * n_tris):
        tris.append(Int(atol(tokens[idx]))); idx += 1

    var grad_m = List[Float64](capacity=n * n)
    var curl_m = List[Float64](capacity=n * n)
    var harm_m = List[Float64](capacity=n * n)
    for _ in range(n * n):
        grad_m.append(0.0)
        curl_m.append(0.0)
        harm_m.append(0.0)

    hodge_decomposition(
        knm, phases, n, edges, n_edges, tris, n_tris, grad_m, curl_m, harm_m
    )
    for k in range(n * n):
        print(grad_m[k])
    for k in range(n * n):
        print(curl_m[k])
    for k in range(n * n):
        print(harm_m[k])
