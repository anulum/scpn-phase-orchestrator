# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Python fallback path tests

from __future__ import annotations

import asyncio
import builtins
import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.coupling import connectome, ei_balance
from scpn_phase_orchestrator.monitor import sleep_staging
from scpn_phase_orchestrator.upde import _ref_kernel, sheaf_engine, sparse_engine
from scpn_phase_orchestrator.visualization import streamer


def test_reference_kernel_fallback_methods_match_manual_dynamics() -> None:
    phases = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    omegas = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    knm = np.array(
        [
            [0.0, 0.3, 0.1],
            [0.2, 0.0, 0.4],
            [0.5, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    alpha = np.array(
        [
            [0.0, 0.01, -0.02],
            [0.03, 0.0, 0.04],
            [-0.01, 0.02, 0.0],
        ],
        dtype=np.float64,
    )

    euler = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.2,
        psi=0.7,
        dt=0.01,
        n_steps=3,
        method="euler",
        n_substeps=2,
        atol=1e-8,
        rtol=1e-6,
    )
    manual = phases.copy()
    for _ in range(3):
        for _ in range(2):
            diff = manual[np.newaxis, :] - manual[:, np.newaxis] - alpha
            deriv = omegas + np.sum(knm * np.sin(diff), axis=1)
            deriv += 0.2 * np.sin(0.7 - manual)
            manual = manual + 0.005 * deriv
        manual %= TWO_PI
    np.testing.assert_allclose(euler, manual, rtol=1e-12, atol=1e-12)

    rk4 = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=2,
        method="rk4",
        n_substeps=3,
        atol=1e-8,
        rtol=1e-6,
    )
    rk45 = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.1,
        psi=0.5,
        dt=0.01,
        n_steps=2,
        method="rk45",
        n_substeps=1,
        atol=1e-8,
        rtol=1e-6,
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))
    assert np.all((rk45 >= 0.0) & (rk45 < TWO_PI))


@pytest.mark.parametrize(
    ("method", "n_substeps", "match"),
    [
        ("bogus", 1, "unknown method"),
        ("euler", 0, "n_substeps"),
    ],
)
def test_reference_kernel_rejects_invalid_configuration(
    method: str,
    n_substeps: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ref_kernel.upde_run_python(
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            0.0,
            0.0,
            0.01,
            1,
            method,
            n_substeps,
            1e-8,
            1e-6,
        )


def test_connectome_python_fallback_preserves_brain_network_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connectome, "_HAS_RUST", False)

    knm = connectome.load_hcp_connectome(16, seed=7)

    assert knm.shape == (16, 16)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(knm), 0.0)
    assert np.all(knm >= 0.0)

    left = knm[:8, :8]
    cross = knm[:8, 8:]
    assert float(left[left > 0.0].mean()) > float(cross[cross > 0.0].mean())

    dmn = [int(f * 8) for f in (0.15, 0.45, 0.65, 0.85)]
    dmn += [node + 8 for node in dmn]
    dmn_weights = knm[np.ix_(dmn, dmn)]
    nonzero_dmn = dmn_weights[dmn_weights > 0.0]
    assert float(nonzero_dmn.mean()) > float(knm[knm > 0.0].mean())


def test_neurolib_hcp_loader_validates_and_slices_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.arange(80 * 80, dtype=np.float64).reshape(80, 80)

    class Dataset:
        def __init__(self, name: str) -> None:
            assert name == "hcp"
            self.Cmat = matrix

    neurolib = types.ModuleType("neurolib")
    utils = types.ModuleType("neurolib.utils")
    load_data = types.ModuleType("neurolib.utils.loadData")
    load_data.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "neurolib", neurolib)
    monkeypatch.setitem(sys.modules, "neurolib.utils", utils)
    monkeypatch.setitem(sys.modules, "neurolib.utils.loadData", load_data)

    loaded = connectome.load_neurolib_hcp(5)

    assert loaded.shape == (5, 5)
    np.testing.assert_allclose(np.diag(loaded), 0.0)
    assert np.all(loaded >= 0.0)
    with pytest.raises(ValueError, match=">= 2"):
        connectome.load_neurolib_hcp(1)
    with pytest.raises(ValueError, match="<= 80"):
        connectome.load_neurolib_hcp(81)


def test_neurolib_hcp_loader_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def block_neurolib(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("neurolib"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_neurolib)

    with pytest.raises(ImportError, match="neurolib is required"):
        connectome.load_neurolib_hcp(2)


def test_sleep_staging_python_and_rust_adapter_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sleep_staging, "_HAS_RUST", False)
    assert sleep_staging.classify_sleep_stage(0.72) == "N3"
    assert sleep_staging.classify_sleep_stage(0.45) == "N2"
    assert sleep_staging.classify_sleep_stage(0.32) == "N1"
    assert sleep_staging.classify_sleep_stage(0.32, functional_desync=True) == "REM"
    assert sleep_staging.classify_sleep_stage(0.22, functional_desync=True) == "REM"
    assert sleep_staging.classify_sleep_stage(0.18, functional_desync=True) == "Wake"

    timestamps = np.array([0.0, 900.0, 1800.0, 2700.0], dtype=np.float64)
    phase = sleep_staging.ultradian_phase(timestamps, ["Wake", "N3", "N2", "REM"])
    assert phase == pytest.approx((2700.0 - 900.0) / (90.0 * 60.0))
    assert sleep_staging.ultradian_phase(np.array([]), []) == 0.0
    with pytest.raises(ValueError, match="same length"):
        sleep_staging.ultradian_phase(timestamps, ["Wake", "N2"])

    if hasattr(sleep_staging, "_rust_classify"):
        monkeypatch.setattr(sleep_staging, "_HAS_RUST", True)
        monkeypatch.setattr(sleep_staging, "_rust_classify", lambda r, d: 4)
        monkeypatch.setattr(sleep_staging, "_rust_ultradian", lambda ts, codes: 0.25)
        assert sleep_staging.classify_sleep_stage(0.1) == "REM"
        assert sleep_staging.ultradian_phase(timestamps[:2], ["N3", "REM"]) == 0.25


def test_ei_balance_python_fallback_handles_degenerate_and_scaling_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ei_balance, "_HAS_RUST", False)
    knm = np.array(
        [
            [0.0, 0.6, 0.6, 0.6],
            [0.6, 0.0, 0.6, 0.6],
            [0.2, 0.2, 0.0, 0.2],
            [0.2, 0.2, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    balance = ei_balance.compute_ei_balance(knm, [0, 1, 99], [2, 3, 99])
    assert balance.ratio == pytest.approx(3.0)
    assert not balance.is_balanced

    adjusted = ei_balance.adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
    adjusted_balance = ei_balance.compute_ei_balance(adjusted, [0, 1], [2, 3])
    assert adjusted_balance.ratio == pytest.approx(1.0)
    np.testing.assert_allclose(adjusted[:2], knm[:2])
    off_diagonal = ~np.eye(4, dtype=bool)[2:]
    assert np.all(adjusted[2:][off_diagonal] > knm[2:][off_diagonal])

    np.testing.assert_allclose(ei_balance.adjust_ei_ratio(knm, [0, 1], [], 1.0), knm)
    np.testing.assert_allclose(
        ei_balance.adjust_ei_ratio(np.zeros((4, 4)), [0, 1], [2, 3], 1.0),
        np.zeros((4, 4)),
    )

    balanced = np.ones((4, 4), dtype=np.float64)
    np.testing.assert_allclose(
        ei_balance.adjust_ei_ratio(balanced, [0, 1], [2, 3], 1.0),
        balanced,
    )


def test_sparse_engine_python_fallback_matches_dense_euler_and_rk4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse_engine, "_HAS_RUST", False)
    row_ptr = np.array([0, 2, 4, 6], dtype=np.int64)
    col_indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    knm_values = np.array([0.3, 0.1, 0.2, 0.4, 0.5, 0.1], dtype=np.float64)
    alpha_values = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02], dtype=np.float64)
    phases = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    omegas = np.array([0.2, -0.1, 0.05], dtype=np.float64)

    engine = sparse_engine.SparseUPDEEngine(3, 0.01, method="euler")
    result = engine.step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.2, 0.7, alpha_values
    )

    deriv = omegas.copy()
    for i in range(3):
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            j = col_indices[idx]
            deriv[i] += knm_values[idx] * np.sin(
                phases[j] - phases[i] - alpha_values[idx]
            )
    deriv += 0.2 * np.sin(0.7 - phases)
    np.testing.assert_allclose(result, (phases + 0.01 * deriv) % TWO_PI)

    zero_step = engine.run(
        phases,
        omegas,
        row_ptr,
        col_indices,
        knm_values,
        0.0,
        0.0,
        alpha_values,
        0,
    )
    assert np.shares_memory(zero_step, phases) is False

    rk4 = sparse_engine.SparseUPDEEngine(3, 0.01, method="rk4").run(
        phases, omegas, row_ptr, col_indices, knm_values, 0.1, 0.5, alpha_values, 2
    )
    rk45 = sparse_engine.SparseUPDEEngine(3, 0.01, method="rk45").step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.1, 0.5, alpha_values
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))


def test_sparse_engine_rejects_invalid_python_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse_engine, "_HAS_RUST", False)
    for kwargs in (
        {"n_oscillators": True, "dt": 0.01},
        {"n_oscillators": 0, "dt": 0.01},
        {"n_oscillators": 2, "dt": False},
        {"n_oscillators": 2, "dt": float("nan")},
    ):
        with pytest.raises(ValueError):
            sparse_engine.SparseUPDEEngine(**kwargs)
    with pytest.raises(ValueError, match="Unknown method"):
        sparse_engine.SparseUPDEEngine(2, 0.01, method="bad")
    engine = sparse_engine.SparseUPDEEngine(2, 0.01)
    with pytest.raises(ValueError, match="n_steps"):
        engine.run(
            np.zeros(2),
            np.zeros(2),
            np.array([0, 0, 0]),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            0.0,
            0.0,
            np.array([], dtype=np.float64),
            -1,
        )


def test_sheaf_engine_python_fallback_respects_restriction_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sheaf_engine, "_HAS_RUST", False)
    phases = np.array([[0.1, 0.4], [0.9, 1.2]], dtype=np.float64)
    omegas = np.array([[0.2, -0.1], [0.05, 0.15]], dtype=np.float64)
    restriction = np.zeros((2, 2, 2, 2), dtype=np.float64)
    restriction[0, 1] = np.array([[0.3, 0.1], [0.2, 0.4]])
    restriction[1, 0] = np.array([[0.5, 0.0], [0.1, 0.2]])
    psi = np.array([0.7, 1.0], dtype=np.float64)

    engine = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="euler")
    result = engine.step(phases, omegas, restriction, 0.2, psi)

    deriv = omegas.copy()
    for i in range(2):
        for dim in range(2):
            for j in range(2):
                for k in range(2):
                    deriv[i, dim] += restriction[i, j, dim, k] * np.sin(
                        phases[j, k] - phases[i, dim]
                    )
            deriv[i, dim] += 0.2 * np.sin(psi[dim] - phases[i, dim])
    np.testing.assert_allclose(result, (phases + 0.01 * deriv) % TWO_PI)

    rk4 = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="rk4").run(
        phases, omegas, restriction, 0.1, psi, 2
    )
    rk45 = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="rk45").step(
        phases, omegas, restriction, 0.1, psi
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))


def test_sheaf_engine_rejects_invalid_python_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sheaf_engine, "_HAS_RUST", False)
    for kwargs in (
        {"n_oscillators": True, "d_dimensions": 2, "dt": 0.01},
        {"n_oscillators": 2, "d_dimensions": 0, "dt": 0.01},
        {"n_oscillators": 2, "d_dimensions": 2, "dt": False},
        {"n_oscillators": 2, "d_dimensions": 2, "dt": float("inf")},
    ):
        with pytest.raises(ValueError):
            sheaf_engine.SheafUPDEEngine(**kwargs)
    with pytest.raises(ValueError, match="Unknown method"):
        sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="bad")
    engine = sheaf_engine.SheafUPDEEngine(2, 2, 0.01)
    with pytest.raises(ValueError, match="n_steps"):
        engine.run(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2, 2, 2)),
            0.0,
            np.zeros(2),
            0,
        )


def test_visualizer_streamer_serializes_numpy_and_broadcasts_to_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    class Client:
        async def send(self, message: str) -> None:
            sent.append(message)

    class Future:
        pass

    loop = asyncio.new_event_loop()

    def fake_submit(coro: object, loop_arg: object) -> Future:
        assert loop_arg is loop
        asyncio.run(coro)
        return Future()

    monkeypatch.setattr(streamer.asyncio, "run_coroutine_threadsafe", fake_submit)
    visualizer = streamer.VisualizerStreamer()
    visualizer._loop = loop
    visualizer._clients.add(Client())

    visualizer.broadcast(
        {
            "phase": np.array([0.1, 0.2], dtype=np.float64),
            "count": np.int64(3),
            "nested": [np.float32(0.5), {"matrix": np.eye(2, dtype=np.float64)}],
        }
    )

    assert len(sent) == 1
    assert sent[0] == (
        '{"phase": [0.1, 0.2], "count": 3, '
        '"nested": [0.5, {"matrix": [[1.0, 0.0], [0.0, 1.0]]}]}'
    )
    loop.close()


def test_visualizer_streamer_lifecycle_error_and_handler_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamer, "HAS_WEBSOCKETS", False)
    with pytest.raises(RuntimeError, match="websockets required"):
        streamer.VisualizerStreamer().start()

    class WebSocket:
        def __init__(self) -> None:
            self.closed = False

        async def wait_closed(self) -> None:
            self.closed = True

    async def exercise_handler() -> None:
        visualizer = streamer.VisualizerStreamer()
        ws = WebSocket()
        await visualizer._handler(ws)
        assert ws.closed
        assert ws not in visualizer._clients

    asyncio.run(exercise_handler())
