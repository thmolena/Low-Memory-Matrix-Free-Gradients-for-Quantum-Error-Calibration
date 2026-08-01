import numpy as np

from rc_krylov_qec.lanczos import hermitian_lanczos


def test_lanczos_relation_with_reorthogonalization() -> None:
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    matrix = 0.5 * (raw + raw.conj().T)
    start = rng.normal(size=10) + 1j * rng.normal(size=10)

    result = hermitian_lanczos(
        lambda vector: matrix @ vector,
        start,
        7,
        reorthogonalize=True,
    )
    basis = result.basis
    residual = matrix @ basis - basis @ result.tridiagonal
    assert result.orthogonality_error < 1e-12
    assert np.linalg.norm(residual[:, :-1]) < 1e-11
    assert np.isclose(
        np.linalg.norm(residual[:, -1]),
        result.residual_beta,
        rtol=1e-10,
        atol=1e-12,
    )


def test_full_dimension_quadrature_is_exact() -> None:
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(8, 8))
    matrix = 0.5 * (raw + raw.T)
    start = rng.normal(size=8)
    result = hermitian_lanczos(
        lambda vector: matrix @ vector,
        start.astype(complex),
        8,
        reorthogonalize=True,
    )
    projected_exponential = np.linalg.eigh(result.tridiagonal)
    values, vectors = projected_exponential
    estimate = (
        result.start_norm**2
        * np.dot(vectors[0, :] ** 2, np.exp(values))
    )
    exact_values, exact_vectors = np.linalg.eigh(matrix)
    coordinates = exact_vectors.T @ start
    exact = np.dot(coordinates**2, np.exp(exact_values))
    np.testing.assert_allclose(estimate, exact, rtol=1e-11, atol=1e-11)

