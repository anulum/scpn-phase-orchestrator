# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo LAPACK FFI sanity test

"""Reference pattern for calling ``liblapack.so.3 :: dgels_`` from
Mojo 0.26.2.

Pattern summary
---------------

* ``std.ffi.OwnedDLHandle(path)`` opens the shared library; the
  constructor ``raises``, so ``main`` must be declared ``raises``.
* ``handle.call["symbol_name", ReturnType](args...)`` resolves
  the symbol and calls it with the given return type. Use
  ``NoneType`` for Fortran subroutines (no return).
* All scalars must be passed by pointer per Fortran calling
  convention. Backing them with size-1 ``List[Int32]`` /
  ``List[Int8]`` and using ``.unsafe_ptr()`` avoids the
  ``UnsafePointer.alloc`` / ``address_of`` origin inference
  problems that bite in Mojo 0.26.

The hand-computed 4×2 least-squares problem

    A = [[1, 2], [3, 4], [5, 6], [7, 8]]   (column-major)
    b = [1.0, 2.0, 2.5, 3.0]

has the closed-form solution x = [-0.5, 0.825]. This test
expects LAPACK to return that within ~1e-14.

Build with::

    mojo build mojo/_lapack_test.mojo -o mojo/_lapack_test
"""

from std.ffi import OwnedDLHandle
from std.collections import List


fn main() raises:
    var lib = OwnedDLHandle(
        "/usr/lib/x86_64-linux-gnu/liblapack.so.3",
    )

    var m = List[Int32](capacity=1); m.append(4)
    var n = List[Int32](capacity=1); n.append(2)
    var nrhs = List[Int32](capacity=1); nrhs.append(1)
    var lda = List[Int32](capacity=1); lda.append(4)
    var ldb = List[Int32](capacity=1); ldb.append(4)
    var lwork = List[Int32](capacity=1); lwork.append(1024)
    var info = List[Int32](capacity=1); info.append(0)
    var trans = List[Int8](capacity=1); trans.append(Int8(ord("N")))

    var A = List[Float64](capacity=8)
    A.append(1.0); A.append(3.0); A.append(5.0); A.append(7.0)
    A.append(2.0); A.append(4.0); A.append(6.0); A.append(8.0)

    var B = List[Float64](capacity=4)
    B.append(1.0); B.append(2.0); B.append(2.5); B.append(3.0)

    var work = List[Float64](capacity=1024)
    for _ in range(1024):
        work.append(0.0)

    _ = lib.call["dgels_", NoneType](
        trans.unsafe_ptr(),
        m.unsafe_ptr(), n.unsafe_ptr(), nrhs.unsafe_ptr(),
        A.unsafe_ptr(), lda.unsafe_ptr(),
        B.unsafe_ptr(), ldb.unsafe_ptr(),
        work.unsafe_ptr(), lwork.unsafe_ptr(),
        info.unsafe_ptr(),
    )

    print("INFO:", info[0])
    print("x[0] =", B[0])
    print("x[1] =", B[1])
