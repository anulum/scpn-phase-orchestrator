# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD-with-control solve (Julia port)

"""
Koopman EDMD-with-control least-squares solve (Korda & Mezić 2018).

Solves the two Tikhonov-regularised normal systems

    (ΦᵀΦ + ρI) [Aᵀ; Bᵀ] = Φᵀ Y_lift,     Φ = [X_lift | U]
    (X_liftᵀ X_lift + ρI) Cᵀ = X_liftᵀ X

through Julia's dense linear solver, matching the NumPy reference to machine
precision. Inputs are row-major snapshot matrices (K samples in the rows).
"""
module KoopmanEdmd

using LinearAlgebra

function koopman_edmd_solve(x_lift, inputs, y_lift, states, regularisation::Real)
    xl = Matrix{Float64}(x_lift)
    ui = Matrix{Float64}(inputs)
    yl = Matrix{Float64}(y_lift)
    st = Matrix{Float64}(states)
    n_lift = size(xl, 2)
    rho = Float64(regularisation)

    phi = hcat(xl, ui)
    gram = phi' * phi + rho * I
    cross = phi' * yl
    m_sol = gram \ cross                 # (N+m) × N = [Aᵀ; Bᵀ]
    a = Matrix(transpose(m_sol[1:n_lift, :]))
    b = Matrix(transpose(m_sol[(n_lift + 1):end, :]))

    lift_gram = xl' * xl + rho * I
    ct = lift_gram \ (xl' * st)          # N × n = Cᵀ
    c = Matrix(transpose(ct))
    return (a, b, c)
end

end  # module KoopmanEdmd
