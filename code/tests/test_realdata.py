"""Network-free tests for the authenticated real-Hamiltonian study.

A small symmetric matrix exercises the same derivative, bound, storage, and
timing code paths used for the downloaded PARSEC matrices.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import expm

from rc_krylov_qec.hamiltonians import (
    PARSEC_MATRICES,
    _region_projectors,
    assemble_problem,
    download_matrix,
)
from rc_krylov_qec.realstudy import (
    _dense_arrays,
    dense_frechet_gradient,
    run_bound_study,
    run_cross_matrix_study,
    run_storage_study,
    run_timing_study,
)


def _synthetic_problem(dimension: int = 48, time: float = 10.0):
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((dimension, dimension))
    symmetric = 0.5 * (dense + dense.T)
    matrix = sp.csr_matrix(symmetric)
    return assemble_problem(
        "synthetic", matrix, sha256="", time=time, probe_seed=1
    )


def test_region_projectors_partition_the_index_set():
    diagonals = _region_projectors(20, 3)
    assert len(diagonals) == 3
    total = np.sum(diagonals, axis=0)
    # Disjoint cover: every index belongs to exactly one region, so each
    # projector has spectral norm one and their sum is the identity diagonal.
    assert np.array_equal(total, np.ones(20))
    for diagonal in diagonals:
        assert set(np.unique(diagonal)).issubset({0.0, 1.0})
        assert diagonal.max() == 1.0


def test_matrix_specs_have_valid_pins():
    for spec in PARSEC_MATRICES.values():
        assert spec.url.startswith("https://")
        assert len(spec.sha256) == 64
        assert spec.dimension > 0


def test_direct_replay_matches_rerun_centered_difference():
    problem = _synthetic_problem(dimension=24, time=1.4)
    rows = run_cross_matrix_study({"synthetic": problem})
    assert len(rows) == 6
    assert max(float(row["direct_replay_error"]) for row in rows) < 2.0e-8
    assert all(
        np.isfinite(float(row["forward_only_error"])) for row in rows
    )


def test_unprojected_bound_covers_dense_gradient_on_study_path():
    problem = _synthetic_problem(dimension=30, time=2.0)
    _, dense_gradient = dense_frechet_gradient(problem)
    rows = run_bound_study(problem, dense_gradient)
    assert rows
    assert all(row["bound_covered"] == 1.0 for row in rows)
    assert all(row["minimum_component_slack"] >= -1e-12 for row in rows)
    assert rows[-1]["gradient_error_bound"] < rows[0]["gradient_error_bound"]


def test_scaling_study_reports_memory_saving():
    dimension = 512
    diagonal = np.linspace(-2.0, 2.0, dimension)
    off_diagonal = np.full(dimension - 1, -0.15)
    matrix = sp.diags(
        (off_diagonal, diagonal, off_diagonal), (-1, 0, 1), format="csr"
    )
    problem = assemble_problem(
        "synthetic", matrix, sha256="", time=1.0, probe_seed=1
    )
    rows = run_storage_study({"synthetic": problem})
    assert len(rows) == 1
    row = rows[0]
    assert row["full_basis_mib"] > row["direct_replay_mib"] > 0.0
    assert row["storage_reduction"] > 1.0
    assert row["hamiltonian_matvecs"] == 336.0
    assert row["derivative_matvecs"] == 144.0


def test_dense_frechet_reference_and_timing_path():
    problem = _synthetic_problem(dimension=18, time=1.3)
    hamiltonian, directions = _dense_arrays(problem)
    probability_frechet, gradient_frechet = dense_frechet_gradient(problem)
    epsilon = 2.0e-6
    probabilities = []
    gradient_finite = np.empty(problem.n_parameters)
    for index, direction in enumerate(directions):
        shifted = []
        for sign in (1.0, -1.0):
            propagator = expm(
                -1j * problem.time * (hamiltonian + sign * epsilon * direction)
            )
            amplitude = np.vdot(problem.probe, propagator @ problem.probe)
            shifted.append(float(abs(amplitude) ** 2))
        gradient_finite[index] = (shifted[0] - shifted[1]) / (2.0 * epsilon)
    propagator = expm(-1j * problem.time * hamiltonian)
    probability_finite = float(
        abs(np.vdot(problem.probe, propagator @ problem.probe)) ** 2
    )
    np.testing.assert_allclose(
        probability_frechet, probability_finite, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient_frechet, gradient_finite, rtol=2e-7, atol=2e-9
    )

    rows = run_timing_study(problem, hamiltonian, directions)
    assert len(rows) == 7
    assert rows[0]["dense_frechet_seconds"] > 0.0
    assert rows[0]["direct_replay_seconds"] > 0.0
    assert rows[0]["gradient_l2_error"] < 1e-8


def test_download_matrix_rejects_checksum_mismatch(tmp_path: Path):
    # Pre-place a bogus archive at the expected cache location so no network
    # access occurs; the pinned digest must then reject it.
    spec = PARSEC_MATRICES["Si2"]
    archive = tmp_path / f"{spec.group}_{spec.name}.tar.gz"
    dummy = tmp_path / "dummy.mtx"
    dummy.write_text("%%MatrixMarket matrix coordinate real symmetric\n1 1 1\n1 1 1.0\n")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(dummy, arcname="Si2/Si2.mtx")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_matrix("Si2", cache_dir=tmp_path, verify=True)
