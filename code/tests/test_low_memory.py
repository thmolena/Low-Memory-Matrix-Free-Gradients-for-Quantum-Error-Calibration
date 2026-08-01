import numpy as np

from rc_krylov_qec.lanczos import hermitian_lanczos
from rc_krylov_qec.low_memory import (
    bessel_coefficient_tail_upper_bound,
    direct_tridiagonal_replay_gradient,
    error_bounded_survival_gradient,
    low_memory_lanczos,
    replay_combinations,
    tangent_tridiagonal_norms,
)
from rc_krylov_qec.pauli import PauliHamiltonian, local_ising_terms
from rc_krylov_qec.projected import survival_probability_gradient
from rc_krylov_qec.reference import dense_survival_probability_gradient


def test_low_memory_coefficients_and_replay_match_stored_basis() -> None:
    rng = np.random.default_rng(101)
    raw = rng.normal(size=(14, 14)) + 1j * rng.normal(size=(14, 14))
    matrix = 0.5 * (raw + raw.conj().T)
    start = rng.normal(size=14) + 1j * rng.normal(size=14)
    steps = 9

    stored = hermitian_lanczos(
        lambda vector: matrix @ vector,
        start,
        steps,
        reorthogonalize=False,
    )
    compact = low_memory_lanczos(
        lambda vector: matrix @ vector,
        start,
        steps,
    )
    np.testing.assert_allclose(
        compact.tridiagonal,
        stored.tridiagonal,
        rtol=2e-13,
        atol=2e-13,
    )

    weights = rng.normal(size=(steps, 3))
    replayed = replay_combinations(
        lambda vector: matrix @ vector,
        compact,
        weights,
    )
    np.testing.assert_allclose(
        replayed,
        stored.basis @ weights,
        rtol=2e-12,
        atol=2e-12,
    )


def test_residual_correction_matches_finite_difference_of_projected_value():
    rng = np.random.default_rng(103)
    terms = local_ising_terms(5)
    parameters = rng.normal(scale=0.22, size=len(terms))
    hamiltonian = PauliHamiltonian(terms, parameters)
    state = rng.normal(size=hamiltonian.dimension) + 1j * rng.normal(
        size=hamiltonian.dimension
    )
    state /= np.linalg.norm(state)
    time = 1.4
    steps = 9

    bounded_result = error_bounded_survival_gradient(
        hamiltonian.matvec,
        state,
        hamiltonian.derivative_matvecs(),
        time,
        steps,
        rank=steps,
    )

    epsilon = 2e-6
    finite_difference = np.empty(len(terms))
    for index in range(len(terms)):
        direction = np.zeros(len(terms))
        direction[index] = epsilon
        plus = hamiltonian.with_parameters(parameters + direction)
        minus = hamiltonian.with_parameters(parameters - direction)
        plus_coefficients = low_memory_lanczos(
            plus.matvec, state, steps
        )
        minus_coefficients = low_memory_lanczos(
            minus.matvec, state, steps
        )
        from rc_krylov_qec.projected import (
            survival_probability_sensitivity,
        )

        plus_probability, _ = survival_probability_sensitivity(
            plus_coefficients.start_norm,
            plus_coefficients.tridiagonal,
            time,
        )
        minus_probability, _ = survival_probability_sensitivity(
            minus_coefficients.start_norm,
            minus_coefficients.tridiagonal,
            time,
        )
        finite_difference[index] = (
            plus_probability - minus_probability
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        bounded_result.gradient,
        finite_difference,
        rtol=3e-6,
        atol=3e-8,
    )


def test_direct_tridiagonal_replay_matches_finite_projected_derivative():
    rng = np.random.default_rng(104)
    terms = local_ising_terms(5)
    parameters = rng.normal(scale=0.2, size=len(terms))
    hamiltonian = PauliHamiltonian(terms, parameters)
    state = rng.normal(size=hamiltonian.dimension) + 1j * rng.normal(
        size=hamiltonian.dimension
    )
    state /= np.linalg.norm(state)
    time = 1.6
    steps = 10

    direct = direct_tridiagonal_replay_gradient(
        hamiltonian.matvec,
        state,
        hamiltonian.derivative_matvecs(),
        time,
        steps,
    )
    full_rank = error_bounded_survival_gradient(
        hamiltonian.matvec,
        state,
        hamiltonian.derivative_matvecs(),
        time,
        steps,
        rank=steps,
    )
    np.testing.assert_allclose(
        direct.probability,
        full_rank.probability,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        direct.gradient,
        full_rank.gradient,
        rtol=2e-11,
        atol=2e-12,
    )
    assert direct.hamiltonian_matvecs == steps * (1 + 2 * len(terms))
    assert direct.derivative_matvecs == steps * len(terms)


def test_direct_tridiagonal_replay_total_bound_covers_dense_gradient():
    rng = np.random.default_rng(105)
    dimension = 14
    raw = rng.normal(size=(dimension, dimension))
    matrix = 0.5 * (raw + raw.T)
    matrix /= np.linalg.norm(matrix, ord=2)
    directions = []
    for _ in range(3):
        raw_direction = rng.normal(size=(dimension, dimension))
        direction = 0.5 * (raw_direction + raw_direction.T)
        direction /= np.linalg.norm(direction, ord=2)
        directions.append(direction.astype(np.complex128))
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    state /= np.linalg.norm(state)
    derivative_actions = [
        lambda vector, direction=direction: direction @ vector
        for direction in directions
    ]
    direct = direct_tridiagonal_replay_gradient(
        lambda vector: matrix @ vector,
        state,
        derivative_actions,
        1.3,
        5,
        derivative_operator_norms=np.ones(3),
        hamiltonian_norm_upper_bound=1.0,
    )
    _, dense_gradient = dense_survival_probability_gradient(
        matrix.astype(np.complex128),
        directions,
        state,
        1.3,
    )
    assert direct.krylov_component_error_bound is not None
    assert direct.krylov_vector_error_bound is not None
    observed = np.abs(direct.gradient - dense_gradient)
    assert np.all(
        observed
        <= direct.krylov_component_error_bound * (1.0 + 2e-10)
        + 2e-12
    )


def test_compression_certificate_covers_forward_gradient_error() -> None:
    rng = np.random.default_rng(107)
    terms = local_ising_terms(6)
    parameters = rng.normal(scale=0.3, size=len(terms))
    hamiltonian = PauliHamiltonian(terms, parameters)
    state = rng.normal(size=hamiltonian.dimension) + 1j * rng.normal(
        size=hamiltonian.dimension
    )
    state /= np.linalg.norm(state)
    time = 2.0
    steps = 14
    rank = 5

    bounded_result = error_bounded_survival_gradient(
        hamiltonian.matvec,
        state,
        hamiltonian.derivative_matvecs(),
        time,
        steps,
        rank,
    )
    stored = hermitian_lanczos(
        hamiltonian.matvec,
        state,
        steps,
        reorthogonalize=False,
    )
    _, uncompressed_forward = survival_probability_gradient(
        stored,
        hamiltonian.derivative_matvecs(),
        time,
    )
    compressed_error = np.abs(
        bounded_result.compressed_forward_gradient - uncompressed_forward
    )
    assert np.all(
        compressed_error
        <= bounded_result.compression_tail_nuclear * (1.0 + 2e-10)
    )


def test_bessel_majorants_cover_scipy_tails() -> None:
    from scipy.special import jv

    for argument, first_degree in ((0.2, 3), (3.0, 8), (12.5, 10)):
        degrees = np.arange(first_degree, first_degree + 500)
        coefficients = np.abs(jv(degrees, argument))
        assert float(np.sum(coefficients)) <= (
            bessel_coefficient_tail_upper_bound(argument, first_degree)
        )
        assert float(np.sum(degrees**2 * coefficients)) <= (
            bessel_coefficient_tail_upper_bound(
                argument, first_degree, degree_squared=True
            )
        )


def test_tangent_tridiagonal_matches_centered_difference() -> None:
    rng = np.random.default_rng(109)
    raw = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    matrix = 0.5 * (raw + raw.conj().T)
    raw_direction = rng.normal(size=(12, 12)) + 1j * rng.normal(
        size=(12, 12)
    )
    direction = 0.5 * (raw_direction + raw_direction.conj().T)
    start = rng.normal(size=12) + 1j * rng.normal(size=12)
    steps = 7
    coefficients = low_memory_lanczos(
        lambda vector: matrix @ vector, start, steps
    )
    tangent_norm = tangent_tridiagonal_norms(
        lambda vector: matrix @ vector,
        coefficients,
        [lambda vector: direction @ vector],
    )[0]
    epsilon = 2.0e-7
    plus = low_memory_lanczos(
        lambda vector: (matrix + epsilon * direction) @ vector,
        start,
        steps,
    ).tridiagonal
    minus = low_memory_lanczos(
        lambda vector: (matrix - epsilon * direction) @ vector,
        start,
        steps,
    ).tridiagonal
    finite_difference_norm = float(
        np.linalg.norm((plus - minus) / (2.0 * epsilon), ord=2)
    )
    np.testing.assert_allclose(
        tangent_norm, finite_difference_norm, rtol=2e-7, atol=2e-8
    )


def test_total_exact_arithmetic_certificate_covers_dense_gradient() -> None:
    rng = np.random.default_rng(113)
    for dimension, steps, time in ((10, 3, 0.8), (14, 5, 1.4), (18, 7, 2.0)):
        raw = rng.normal(size=(dimension, dimension))
        matrix = 0.5 * (raw + raw.T)
        matrix /= np.linalg.norm(matrix, ord=2)
        directions = []
        for _ in range(3):
            raw_direction = rng.normal(size=(dimension, dimension))
            direction = 0.5 * (raw_direction + raw_direction.T)
            direction /= np.linalg.norm(direction, ord=2)
            directions.append(direction.astype(np.complex128))
        start = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        start /= np.linalg.norm(start)
        derivative_matvecs = [
            lambda vector, direction=direction: direction @ vector
            for direction in directions
        ]
        result = error_bounded_survival_gradient(
            lambda vector: matrix @ vector,
            start,
            derivative_matvecs,
            time,
            steps,
            rank=max(1, steps - 2),
            derivative_operator_norms=np.ones(3),
            hamiltonian_norm_upper_bound=1.0,
        )
        _, dense_gradient = dense_survival_probability_gradient(
            matrix.astype(np.complex128),
            directions,
            start,
            time,
        )
        observed = np.abs(result.gradient - dense_gradient)
        assert result.total_component_error_bound is not None
        assert np.all(
            observed
            <= result.total_component_error_bound
            * (1.0 + 2e-10)
            + 2e-12
        )
