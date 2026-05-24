# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral graph eigendecomposition (Mojo port)

"""Symmetric eigendecomposition of the graph Laplacian as a Mojo
executable, backed by LAPACK ``dsyev_`` via the ``std.ffi``
FFI pattern validated in ``_lapack_test.mojo``.

Stdin dispatch:

    EIG n knm[0..n*n]
        → prints ``n`` eigenvalues followed by ``n`` Fiedler-vector
          entries (one per line).

Build with::

    mojo build mojo/spectral.mojo -o mojo/spectral_mojo
"""

from std.ffi import OwnedDLHandle
from std.collections import List


fn spectral_eig(
    knm_flat: List[Float64],
    n: Int,
    mut out_eigvals: List[Float64],
    mut out_fiedler: List[Float64],
) raises -> Int:
    # Build L = D − A where A_ij is the reciprocal undirected
    # magnitude weight (|W_ij| + |W_ji|) / 2 and A_ii = 0.
    # LAPACK ``dsyev_`` expects column-major; since L is symmetric,
    # row-major and column-major are equivalent.
    var L = List[Float64](capacity=n * n)
    for _ in range(n * n):
        L.append(0.0)
    var degree = List[Float64](capacity=n)
    for i in range(n):
        degree.append(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            var w = 0.5 * (abs(knm_flat[i * n + j]) + abs(knm_flat[j * n + i]))
            L[i * n + j] = -w
            L[j * n + i] = -w
            degree[i] += w
            degree[j] += w
    for i in range(n):
        L[i * n + i] = degree[i]

    # LAPACK FFI (pattern validated in _lapack_test.mojo).
    var lib = OwnedDLHandle(
        "/usr/lib/x86_64-linux-gnu/liblapack.so.3",
    )

    var jobz = List[Int8](capacity=1); jobz.append(Int8(ord("V")))
    var uplo = List[Int8](capacity=1); uplo.append(Int8(ord("U")))
    var n_c = List[Int32](capacity=1); n_c.append(Int32(n))
    var lda = List[Int32](capacity=1); lda.append(Int32(n))
    var lwork_len = 3 * n * n + 64
    var lwork = List[Int32](capacity=1); lwork.append(Int32(lwork_len))
    var info = List[Int32](capacity=1); info.append(0)

    var W = List[Float64](capacity=n)
    for _ in range(n):
        W.append(0.0)

    var work = List[Float64](capacity=lwork_len)
    for _ in range(lwork_len):
        work.append(0.0)

    # ``dsyev_`` overwrites the upper (or lower) triangle of A with
    # eigenvectors when JOBZ == 'V'; W receives eigenvalues ascending.
    _ = lib.call["dsyev_", NoneType](
        jobz.unsafe_ptr(),
        uplo.unsafe_ptr(),
        n_c.unsafe_ptr(),
        L.unsafe_ptr(),
        lda.unsafe_ptr(),
        W.unsafe_ptr(),
        work.unsafe_ptr(),
        lwork.unsafe_ptr(),
        info.unsafe_ptr(),
    )

    if Int(info[0]) != 0:
        return Int(info[0])

    for i in range(n):
        out_eigvals[i] = W[i]
    # Eigenvectors are stored in the columns of A (column-major).
    # Column index 1 = second smallest eigenvalue = Fiedler vector.
    var col = 1
    for row in range(n):
        out_fiedler[row] = L[col * n + row]
    return 0


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "EIG":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1

    var eigvals = List[Float64](capacity=n)
    var fiedler = List[Float64](capacity=n)
    for _ in range(n):
        eigvals.append(0.0)
        fiedler.append(0.0)

    var rc = spectral_eig(knm, n, eigvals, fiedler)
    if rc != 0:
        print("ERR:", rc)
        return
    for i in range(n):
        print(eigvals[i])
    for i in range(n):
        print(fiedler[i])
