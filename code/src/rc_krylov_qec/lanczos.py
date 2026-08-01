"""Complex-Hermitian Lanczos iteration with explicit diagnostic metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

ComplexVector = NDArray[np.complex128]
Matvec = Callable[[ComplexVector], ComplexVector]


@dataclass(frozen=True)
class LanczosResult:
    """Outputs of an ``m``-step Hermitian Lanczos calculation."""

    basis: NDArray[np.complex128]
    tridiagonal: NDArray[np.float64]
    residual_beta: float
    start_norm: float
    orthogonality_error: float

    @property
    def steps(self) -> int:
        return self.tridiagonal.shape[0]


def hermitian_lanczos(
    matvec: Matvec,
    start: ComplexVector,
    steps: int,
    *,
    reorthogonalize: bool = False,
    breakdown_tolerance: float = 1e-14,
) -> LanczosResult:
    """Run Lanczos using only ``matvec`` evaluations.

    When requested, two modified-Gram-Schmidt passes are used for validation.
    The default short recurrence is the algorithm relevant to the proposed
    forward-only gradient estimator.
    """

    initial = np.asarray(start, dtype=np.complex128)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("start must be a nonempty one-dimensional vector")
    if steps < 1:
        raise ValueError("steps must be positive")

    start_norm = float(np.linalg.norm(initial))
    if not np.isfinite(start_norm) or start_norm == 0.0:
        raise ValueError("start must have finite, nonzero norm")

    maximum_steps = min(steps, initial.size)
    basis_vectors: list[ComplexVector] = [initial / start_norm]
    alphas: list[float] = []
    off_diagonals: list[float] = []
    previous = np.zeros_like(initial)
    beta_previous = 0.0
    residual_beta = 0.0

    for iteration in range(maximum_steps):
        current = basis_vectors[iteration]
        work = np.asarray(matvec(current), dtype=np.complex128)
        if work.shape != current.shape:
            raise ValueError("matvec returned a vector with the wrong shape")

        if iteration:
            work -= beta_previous * previous
        alpha_complex = np.vdot(current, work)
        hermitian_defect = abs(alpha_complex.imag)
        tolerance = 100 * np.finfo(float).eps * max(1.0, abs(alpha_complex))
        if hermitian_defect > tolerance:
            raise ValueError("matvec does not appear Hermitian")
        alpha = float(alpha_complex.real)
        work -= alpha * current

        if reorthogonalize:
            stacked = np.column_stack(basis_vectors)
            for _ in range(2):
                work -= stacked @ (stacked.conj().T @ work)

        beta = float(np.linalg.norm(work))
        alphas.append(alpha)
        residual_beta = beta

        if iteration == maximum_steps - 1 or beta <= breakdown_tolerance:
            break

        off_diagonals.append(beta)
        previous = current
        beta_previous = beta
        basis_vectors.append(work / beta)

    basis = np.column_stack(basis_vectors)
    actual_steps = len(alphas)
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if actual_steps > 1:
        off = np.asarray(off_diagonals[: actual_steps - 1], dtype=np.float64)
        tridiagonal += np.diag(off, 1) + np.diag(off, -1)

    identity = np.eye(actual_steps)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        orthogonality_error = float(
            np.linalg.norm(basis.conj().T @ basis - identity, ord="fro")
        )
    if not np.isfinite(orthogonality_error):
        raise FloatingPointError("Lanczos orthogonality diagnostic is nonfinite")
    return LanczosResult(
        basis=basis,
        tridiagonal=tridiagonal,
        residual_beta=residual_beta,
        start_norm=start_norm,
        orthogonality_error=orthogonality_error,
    )
