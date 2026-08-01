"""Projected matrix-function values and forward-only parameter sensitivities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .lanczos import LanczosResult

ScalarFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
DerivativeMatvec = Callable[
    [NDArray[np.complex128]], NDArray[np.complex128]
]


def projected_quadratic_form(
    lanczos: LanczosResult,
    function: ScalarFunction,
) -> float:
    """Evaluate ``||u||^2 e1.T f(T) e1`` for a real scalar function."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eigenvalues, eigenvectors = np.linalg.eigh(lanczos.tridiagonal)
        weights = eigenvectors[0, :] ** 2
        value = float(
            lanczos.start_norm**2 * np.dot(weights, function(eigenvalues))
        )
    if not np.isfinite(value):
        raise FloatingPointError("projected quadratic form is nonfinite")
    return value


def divided_difference(
    eigenvalues: NDArray[np.float64],
    function: ScalarFunction,
    derivative: ScalarFunction,
) -> NDArray[np.float64]:
    """Stable first divided-difference matrix."""

    values = function(eigenvalues)
    numerator = values[:, None] - values[None, :]
    denominator = eigenvalues[:, None] - eigenvalues[None, :]
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    close = np.abs(denominator) <= 32 * np.finfo(float).eps * scale

    result = np.empty_like(denominator)
    np.divide(numerator, denominator, out=result, where=~close)
    midpoint = 0.5 * (eigenvalues[:, None] + eigenvalues[None, :])
    result[close] = derivative(midpoint[close])
    return result


def projected_sensitivity(
    lanczos: LanczosResult,
    function: ScalarFunction,
    derivative: ScalarFunction,
) -> NDArray[np.float64]:
    """Return the sensitivity of the small projected quadratic form."""

    eigenvalues, eigenvectors = np.linalg.eigh(lanczos.tridiagonal)
    c = eigenvectors.T[:, 0]
    frechet_kernel = divided_difference(eigenvalues, function, derivative)
    spectral_sensitivity = np.outer(c, c) * frechet_kernel
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sensitivity = (
            lanczos.start_norm**2
            * eigenvectors
            @ spectral_sensitivity
            @ eigenvectors.T
        )
        sensitivity = np.asarray(
            0.5 * (sensitivity + sensitivity.T), dtype=np.float64
        )
    if not np.all(np.isfinite(sensitivity)):
        raise FloatingPointError("projected sensitivity is nonfinite")
    return sensitivity


def contract_parameter_gradients(
    lanczos: LanczosResult,
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> NDArray[np.float64]:
    """Contract projected sensitivity with matrix-free parameter derivatives."""

    basis = lanczos.basis
    if sensitivity.shape != (lanczos.steps, lanczos.steps):
        raise ValueError("sensitivity shape does not match the Lanczos basis")

    gradients = np.empty(len(derivative_matvecs), dtype=np.float64)
    for index, derivative_matvec in enumerate(derivative_matvecs):
        applied = np.column_stack(
            [derivative_matvec(basis[:, column]) for column in range(lanczos.steps)]
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            projected_derivative = basis.conj().T @ applied
            gradients[index] = float(
                np.real(np.sum(sensitivity * projected_derivative.T))
            )
    if not np.all(np.isfinite(gradients)):
        raise FloatingPointError("projected parameter gradient is nonfinite")
    return gradients


def survival_probability_gradient(
    lanczos: LanczosResult,
    derivative_matvecs: Sequence[DerivativeMatvec],
    time: float,
) -> tuple[float, NDArray[np.float64]]:
    """Approximate ``|u^dagger exp(-itA)u|^2`` and its parameter gradient."""

    cosine = lambda x: np.cos(time * x)
    cosine_derivative = lambda x: -time * np.sin(time * x)
    sine = lambda x: np.sin(time * x)
    sine_derivative = lambda x: time * np.cos(time * x)

    real_amplitude = projected_quadratic_form(lanczos, cosine)
    minus_imaginary_amplitude = projected_quadratic_form(lanczos, sine)
    cosine_sensitivity = projected_sensitivity(
        lanczos, cosine, cosine_derivative
    )
    sine_sensitivity = projected_sensitivity(lanczos, sine, sine_derivative)
    cosine_gradient = contract_parameter_gradients(
        lanczos, cosine_sensitivity, derivative_matvecs
    )
    sine_gradient = contract_parameter_gradients(
        lanczos, sine_sensitivity, derivative_matvecs
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        probability = real_amplitude**2 + minus_imaginary_amplitude**2
        gradient = (
            2.0 * real_amplitude * cosine_gradient
            + 2.0 * minus_imaginary_amplitude * sine_gradient
        )
    if not np.isfinite(probability) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("projected survival gradient is nonfinite")
    return float(probability), gradient


def survival_probability_sensitivity(
    start_norm: float,
    tridiagonal: NDArray[np.float64],
    time: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the projected survival probability and its sensitivity to ``T``.

    The returned matrix ``G`` satisfies
    ``d p_hat = trace(G.T @ dT)`` for the projected probability
    ``p_hat = | ||u||^2 e1.T exp(-i t T) e1 |^2``.
    """

    projected_result = LanczosResult(
        basis=np.empty((0, tridiagonal.shape[0]), dtype=np.complex128),
        tridiagonal=np.asarray(tridiagonal, dtype=np.float64),
        residual_beta=0.0,
        start_norm=float(start_norm),
        orthogonality_error=float("nan"),
    )
    cosine = lambda x: np.cos(time * x)
    cosine_derivative = lambda x: -time * np.sin(time * x)
    sine = lambda x: np.sin(time * x)
    sine_derivative = lambda x: time * np.cos(time * x)

    real_amplitude = projected_quadratic_form(projected_result, cosine)
    minus_imaginary_amplitude = projected_quadratic_form(
        projected_result, sine
    )
    cosine_sensitivity = projected_sensitivity(
        projected_result, cosine, cosine_derivative
    )
    sine_sensitivity = projected_sensitivity(
        projected_result, sine, sine_derivative
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        probability = real_amplitude**2 + minus_imaginary_amplitude**2
        sensitivity = (
            2.0 * real_amplitude * cosine_sensitivity
            + 2.0 * minus_imaginary_amplitude * sine_sensitivity
        )
    if not np.isfinite(probability) or not np.all(np.isfinite(sensitivity)):
        raise FloatingPointError("projected survival sensitivity is nonfinite")
    return float(probability), sensitivity
