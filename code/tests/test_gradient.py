import numpy as np

from rc_krylov_qec.lanczos import hermitian_lanczos
from rc_krylov_qec.low_memory import (
    low_memory_survival_probability_gradient,
)
from rc_krylov_qec.pauli import PauliHamiltonian, local_ising_terms
from rc_krylov_qec.projected import survival_probability_gradient
from rc_krylov_qec.reference import dense_survival_probability_gradient


def test_projected_gradient_converges_to_dense_reference() -> None:
    rng = np.random.default_rng(31)
    terms = local_ising_terms(4)
    hamiltonian = PauliHamiltonian(
        terms,
        rng.normal(scale=0.25, size=len(terms)),
    )
    state = rng.normal(size=hamiltonian.dimension) + 1j * rng.normal(
        size=hamiltonian.dimension
    )
    state /= np.linalg.norm(state)
    time = 1.7

    reference_probability, reference_gradient = (
        dense_survival_probability_gradient(
            hamiltonian.dense(),
            [term.dense() for term in terms],
            state,
            time,
        )
    )
    result = hermitian_lanczos(
        hamiltonian.matvec,
        state,
        hamiltonian.dimension,
        reorthogonalize=True,
    )
    probability, gradient = survival_probability_gradient(
        result,
        hamiltonian.derivative_matvecs(),
        time,
    )
    np.testing.assert_allclose(
        probability, reference_probability, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=2e-9, atol=2e-10
    )


def test_full_rank_two_pass_gradient_matches_full_basis() -> None:
    rng = np.random.default_rng(41)
    terms = local_ising_terms(4)
    hamiltonian = PauliHamiltonian(
        terms,
        rng.normal(scale=0.2, size=len(terms)),
    )
    state = rng.normal(size=hamiltonian.dimension) + 1j * rng.normal(
        size=hamiltonian.dimension
    )
    state /= np.linalg.norm(state)
    time = 1.2
    steps = 12

    full_result = hermitian_lanczos(
        hamiltonian.matvec, state, steps, reorthogonalize=False
    )
    full_probability, full_gradient = survival_probability_gradient(
        full_result, hamiltonian.derivative_matvecs(), time
    )
    replay_probability, replay_gradient, diagnostics = (
        low_memory_survival_probability_gradient(
            hamiltonian.matvec,
            state,
            hamiltonian.derivative_matvecs(),
            time,
            steps,
            rank=steps,
        )
    )
    np.testing.assert_allclose(
        replay_probability, full_probability, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        replay_gradient, full_gradient, rtol=1e-10, atol=1e-11
    )
    assert diagnostics.shared_rank == steps
