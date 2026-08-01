import numpy as np

from rc_krylov_qec import (
    direct_tridiagonal_replay_gradient,
    parameter_blocked_replay_gradient,
)


def test_parameter_blocks_equal_scalar_direct_replay() -> None:
    rng = np.random.default_rng(260801)
    matrix = rng.standard_normal((18, 18))
    matrix = 0.5 * (matrix + matrix.T)
    directions = []
    for _ in range(5):
        direction = rng.standard_normal((18, 18))
        directions.append(0.5 * (direction + direction.T))
    start = rng.standard_normal(18).astype(np.complex128)
    matvec = lambda block: matrix @ block
    derivatives = tuple(
        (lambda block, direction=direction: direction @ block)
        for direction in directions
    )
    expected = direct_tridiagonal_replay_gradient(
        matvec, start, derivatives, time=0.3, steps=10
    )
    for width in (1, 2, 5, 9):
        observed = parameter_blocked_replay_gradient(
            matvec,
            start,
            derivatives,
            time=0.3,
            steps=10,
            parameter_block_size=width,
        )
        np.testing.assert_allclose(observed.probability, expected.probability)
        np.testing.assert_allclose(
            observed.gradient, expected.gradient, rtol=2e-12, atol=2e-12
        )
        np.testing.assert_allclose(
            observed.tangent_tridiagonal_norms,
            expected.tangent_tridiagonal_norms,
            rtol=2e-12,
            atol=2e-12,
        )


def test_operator_call_and_storage_ledger() -> None:
    diagonal = np.linspace(-1.0, 1.0, 16)
    matvec = lambda block: diagonal[:, None] * block if block.ndim == 2 else diagonal * block
    derivatives = tuple(
        lambda block, index=index: (
            np.eye(16)[index][:, None] * block
            if block.ndim == 2
            else np.eye(16)[index] * block
        )
        for index in range(5)
    )
    result = parameter_blocked_replay_gradient(
        matvec,
        np.ones(16, dtype=np.complex128),
        derivatives,
        time=0.2,
        steps=8,
        parameter_block_size=2,
    )
    assert result.hamiltonian_vector_calls == 32
    assert result.hamiltonian_block_calls == 24
    assert result.hamiltonian_column_equivalents == 56
    assert result.derivative_vector_calls == 40
    assert result.peak_ambient_vector_equivalents == 14
