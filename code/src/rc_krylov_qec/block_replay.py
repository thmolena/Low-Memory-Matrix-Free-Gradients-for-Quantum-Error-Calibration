"""Parameter-blocked tangent replay for finite Lanczos gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .low_memory import low_memory_lanczos
from .projected import DerivativeMatvec, survival_probability_sensitivity

ComplexArray = NDArray[np.complex128]
Matvec = Callable[[ComplexArray], ComplexArray]


@dataclass(frozen=True)
class BlockReplayResult:
    probability: float
    gradient: NDArray[np.float64]
    tangent_tridiagonal_norms: NDArray[np.float64]
    steps: int
    parameter_count: int
    parameter_block_size: int
    hamiltonian_vector_calls: int
    hamiltonian_block_calls: int
    hamiltonian_column_equivalents: int
    derivative_vector_calls: int
    peak_ambient_vector_equivalents: int


def _tangent_block(
    matvec: Matvec,
    start_vector: ComplexArray,
    alphas: NDArray[np.float64],
    betas: NDArray[np.float64],
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay one parameter block and return exact tridiagonal contractions."""

    width = len(derivative_matvecs)
    steps = alphas.size
    previous = np.zeros_like(start_vector)
    current = start_vector.copy()
    tangent_previous = np.zeros((start_vector.size, width), dtype=np.complex128)
    tangent_current = np.zeros_like(tangent_previous)
    beta_previous = 0.0
    tangent_beta_previous = np.zeros(width, dtype=np.float64)
    tangent_alphas = np.empty((width, steps), dtype=np.float64)
    tangent_betas = np.empty((width, max(0, steps - 1)), dtype=np.float64)

    for iteration in range(steps):
        applied_current = np.asarray(matvec(current), dtype=np.complex128)
        applied_tangent = np.asarray(matvec(tangent_current), dtype=np.complex128)
        if applied_tangent.shape != tangent_current.shape:
            raise ValueError("matvec must preserve a two-dimensional parameter block")
        derivative_current = np.column_stack(
            [
                np.asarray(action(current), dtype=np.complex128)
                for action in derivative_matvecs
            ]
        )
        alpha = float(alphas[iteration])
        beta = float(betas[iteration])
        tangent_alpha = np.real(
            np.sum(np.conjugate(tangent_current) * applied_current[:, None], axis=0)
            + np.sum(np.conjugate(current)[:, None] * derivative_current, axis=0)
            + np.sum(np.conjugate(current)[:, None] * applied_tangent, axis=0)
        )
        tangent_alphas[:, iteration] = tangent_alpha
        if iteration == steps - 1:
            break
        if beta == 0.0:
            raise RuntimeError("cannot replay after Lanczos breakdown")

        residual = applied_current - alpha * current - beta_previous * previous
        next_vector = residual / beta
        tangent_residual = (
            derivative_current
            + applied_tangent
            - current[:, None] * tangent_alpha[None, :]
            - alpha * tangent_current
            - previous[:, None] * tangent_beta_previous[None, :]
            - beta_previous * tangent_previous
        )
        tangent_beta = np.real(
            np.sum(np.conjugate(next_vector)[:, None] * tangent_residual, axis=0)
        )
        tangent_betas[:, iteration] = tangent_beta
        tangent_next = (
            tangent_residual - next_vector[:, None] * tangent_beta[None, :]
        ) / beta

        previous, current = current, next_vector
        tangent_previous, tangent_current = tangent_current, tangent_next
        beta_previous = beta
        tangent_beta_previous = tangent_beta

    gradients = np.empty(width, dtype=np.float64)
    tangent_norms = np.empty(width, dtype=np.float64)
    for column in range(width):
        tangent_matrix = np.diag(tangent_alphas[column])
        if steps > 1:
            tangent_matrix += np.diag(tangent_betas[column], 1)
            tangent_matrix += np.diag(tangent_betas[column], -1)
        gradients[column] = float(np.sum(sensitivity * tangent_matrix))
        tangent_norms[column] = float(np.linalg.norm(tangent_matrix, ord=2))
    return gradients, tangent_norms


def parameter_blocked_replay_gradient(
    matvec: Matvec,
    start: ComplexArray,
    derivative_matvecs: Sequence[DerivativeMatvec],
    time: float,
    steps: int,
    *,
    parameter_block_size: int,
) -> BlockReplayResult:
    """Differentiate the finite Lanczos objective in parameter blocks.

    The operator action must accept both vectors of shape ``(n,)`` and blocks
    of shape ``(n, q)``.  A block width of one recovers scalar tangent replay;
    a block width equal to the parameter count minimizes Hamiltonian launches.
    """

    parameter_count = len(derivative_matvecs)
    if parameter_count < 1:
        raise ValueError("at least one derivative action is required")
    if parameter_block_size < 1:
        raise ValueError("parameter_block_size must be positive")
    width = min(int(parameter_block_size), parameter_count)
    coefficients = low_memory_lanczos(matvec, start, steps)
    probability, sensitivity = survival_probability_sensitivity(
        coefficients.start_norm,
        coefficients.tridiagonal,
        time,
    )
    gradient = np.empty(parameter_count, dtype=np.float64)
    tangent_norms = np.empty(parameter_count, dtype=np.float64)
    blocks = 0
    for first in range(0, parameter_count, width):
        last = min(first + width, parameter_count)
        block_gradient, block_norms = _tangent_block(
            matvec,
            coefficients.start_vector,
            coefficients.alphas,
            coefficients.betas,
            sensitivity,
            derivative_matvecs[first:last],
        )
        gradient[first:last] = block_gradient
        tangent_norms[first:last] = block_norms
        blocks += 1
    arrays = (gradient, tangent_norms)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("parameter-blocked replay is nonfinite")
    return BlockReplayResult(
        probability=float(probability),
        gradient=gradient,
        tangent_tridiagonal_norms=tangent_norms,
        steps=coefficients.steps,
        parameter_count=parameter_count,
        parameter_block_size=width,
        hamiltonian_vector_calls=coefficients.steps * (1 + blocks),
        hamiltonian_block_calls=coefficients.steps * blocks,
        hamiltonian_column_equivalents=coefficients.steps * (2 + parameter_count),
        derivative_vector_calls=coefficients.steps * parameter_count,
        peak_ambient_vector_equivalents=6 + 4 * width,
    )
