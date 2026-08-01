"""Replay-based low-memory projected gradients and residual certificates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .projected import (
    DerivativeMatvec,
    divided_difference,
    survival_probability_sensitivity,
)

ComplexVector = NDArray[np.complex128]
Matvec = Callable[[ComplexVector], ComplexVector]


@dataclass(frozen=True)
class LanczosCoefficients:
    """Short-recurrence data retained by a low-memory Lanczos pass."""

    start_vector: ComplexVector
    start_norm: float
    alphas: NDArray[np.float64]
    betas: NDArray[np.float64]
    residual_vector: Optional[ComplexVector]

    @property
    def steps(self) -> int:
        return self.alphas.size

    @property
    def residual_beta(self) -> float:
        return float(self.betas[-1])

    @property
    def tridiagonal(self) -> NDArray[np.float64]:
        matrix = np.diag(self.alphas)
        if self.steps > 1:
            off = self.betas[: self.steps - 1]
            matrix += np.diag(off, 1) + np.diag(off, -1)
        return matrix


@dataclass(frozen=True)
class GradientResult:
    """A compressed gradient and certificate for the projected objective."""

    probability: float
    gradient: NDArray[np.float64]
    compressed_forward_gradient: NDArray[np.float64]
    residual_correction: NDArray[np.float64]
    component_error_bound: NDArray[np.float64]
    vector_error_bound: float
    compression_tail_nuclear: float
    rank: int
    steps: int
    estimated_peak_vector_equivalents: float
    krylov_component_error_bound: NDArray[np.float64] | None = None
    total_component_error_bound: NDArray[np.float64] | None = None
    total_vector_error_bound: float | None = None
    tangent_tridiagonal_norms: NDArray[np.float64] | None = None
    chebyshev_value_tail: float | None = None
    chebyshev_lipschitz_tail: float | None = None


@dataclass(frozen=True)
class DirectReplayResult:
    """Exact finite-Lanczos gradient from tridiagonal-sensitivity replay.

    Exactness refers to the differentiated finite projected objective in exact
    arithmetic.  When a verified operator-norm bound is supplied, the optional
    error fields also bound the difference from the unprojected survival
    gradient under the assumptions documented by
    :func:`direct_tridiagonal_replay_gradient`.
    """

    probability: float
    gradient: NDArray[np.float64]
    steps: int
    estimated_peak_vector_equivalents: float
    hamiltonian_matvecs: int
    derivative_matvecs: int
    tangent_tridiagonal_norms: NDArray[np.float64]
    krylov_component_error_bound: NDArray[np.float64] | None = None
    krylov_vector_error_bound: float | None = None
    chebyshev_value_tail: float | None = None
    chebyshev_lipschitz_tail: float | None = None


def _finite_public_result(function):
    """Contain platform BLAS flags and reject nonfinite public results."""

    @wraps(function)
    def guarded(*args, **kwargs):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = function(*args, **kwargs)
        if isinstance(result, GradientResult):
            arrays = [
                result.gradient,
                result.compressed_forward_gradient,
                result.residual_correction,
                result.component_error_bound,
            ]
            arrays.extend(
                value
                for value in (
                    result.krylov_component_error_bound,
                    result.total_component_error_bound,
                    result.tangent_tridiagonal_norms,
                )
                if value is not None
            )
            scalars = [
                result.probability,
                result.vector_error_bound,
                result.compression_tail_nuclear,
                result.estimated_peak_vector_equivalents,
            ]
            scalars.extend(
                value
                for value in (
                    result.total_vector_error_bound,
                    result.chebyshev_value_tail,
                    result.chebyshev_lipschitz_tail,
                )
                if value is not None
            )
            if not all(np.all(np.isfinite(value)) for value in arrays) or not all(
                np.isfinite(value) for value in scalars
            ):
                raise FloatingPointError("gradient with error bound is nonfinite")
        elif isinstance(result, DirectReplayResult):
            arrays = [result.gradient, result.tangent_tridiagonal_norms]
            if result.krylov_component_error_bound is not None:
                arrays.append(result.krylov_component_error_bound)
            scalars = [
                result.probability,
                result.estimated_peak_vector_equivalents,
            ]
            scalars.extend(
                value
                for value in (
                    result.krylov_vector_error_bound,
                    result.chebyshev_value_tail,
                    result.chebyshev_lipschitz_tail,
                )
                if value is not None
            )
            if not all(np.all(np.isfinite(value)) for value in arrays) or not all(
                np.isfinite(value) for value in scalars
            ):
                raise FloatingPointError("direct replay result is nonfinite")
        elif isinstance(result, tuple) and len(result) == 3:
            probability, gradient, diagnostics = result
            diagnostic_values = (
                diagnostics.cosine_tail_frobenius,
                diagnostics.sine_tail_frobenius,
                diagnostics.residual_beta,
            )
            if (
                not np.isfinite(probability)
                or not np.all(np.isfinite(gradient))
                or not all(np.isfinite(value) for value in diagnostic_values)
            ):
                raise FloatingPointError("low-memory gradient is nonfinite")
        return result

    return guarded


def low_memory_lanczos(
    matvec: Matvec,
    start: ComplexVector,
    steps: int,
    *,
    breakdown_tolerance: float = 1e-14,
) -> LanczosCoefficients:
    """Run unreorthogonalized Lanczos while retaining no Krylov basis."""

    initial = np.asarray(start, dtype=np.complex128)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("start must be a nonempty one-dimensional vector")
    if steps < 1:
        raise ValueError("steps must be positive")

    start_norm = float(np.linalg.norm(initial))
    if not np.isfinite(start_norm) or start_norm == 0.0:
        raise ValueError("start must have finite, nonzero norm")

    maximum_steps = min(steps, initial.size)
    normalized_start = initial / start_norm
    previous = np.zeros_like(normalized_start)
    current = normalized_start.copy()
    beta_previous = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    residual_vector: Optional[ComplexVector] = None

    for iteration in range(maximum_steps):
        work = np.asarray(matvec(current), dtype=np.complex128)
        if work.shape != current.shape:
            raise ValueError("matvec returned a vector with the wrong shape")
        if iteration:
            work -= beta_previous * previous

        alpha_complex = np.vdot(current, work)
        tolerance = 100 * np.finfo(float).eps * max(
            1.0, abs(alpha_complex)
        )
        if abs(alpha_complex.imag) > tolerance:
            raise ValueError("matvec does not appear Hermitian")
        alpha = float(alpha_complex.real)
        work -= alpha * current
        beta = float(np.linalg.norm(work))
        alphas.append(alpha)
        betas.append(beta)

        if beta <= breakdown_tolerance:
            residual_vector = None
            break
        residual_vector = work / beta
        if iteration == maximum_steps - 1:
            break

        previous, current = current, residual_vector
        beta_previous = beta

    return LanczosCoefficients(
        start_vector=normalized_start,
        start_norm=start_norm,
        alphas=np.asarray(alphas, dtype=np.float64),
        betas=np.asarray(betas, dtype=np.float64),
        residual_vector=residual_vector,
    )


def replay_combinations(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    coefficients: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Form ``V @ coefficients`` by replaying the short recurrence."""

    weights = np.asarray(coefficients, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape[0] != lanczos.steps:
        raise ValueError("coefficients must have one row per Lanczos step")

    combinations = np.zeros(
        (lanczos.start_vector.size, weights.shape[1]),
        dtype=np.complex128,
    )
    previous = np.zeros_like(lanczos.start_vector)
    current = lanczos.start_vector.copy()
    beta_previous = 0.0

    for iteration in range(lanczos.steps):
        combinations += current[:, None] * weights[iteration][None, :]
        if iteration == lanczos.steps - 1:
            break
        work = np.asarray(matvec(current), dtype=np.complex128)
        work -= beta_previous * previous
        work -= lanczos.alphas[iteration] * current
        beta = lanczos.betas[iteration]
        if beta == 0.0:
            raise RuntimeError("cannot replay after Lanczos breakdown")
        previous, current = current, work / beta
        beta_previous = beta
    return combinations


def _truncate_symmetric(
    sensitivity: NDArray[np.float64],
    rank: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    if rank < 1 or rank > sensitivity.shape[0]:
        raise ValueError("rank must be between one and the Krylov dimension")
    eigenvalues, eigenvectors = np.linalg.eigh(sensitivity)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    kept = order[:rank]
    discarded = order[rank:]
    tail = float(np.sum(np.abs(eigenvalues[discarded])))
    return eigenvalues[kept], eigenvectors[:, kept], tail


def compressed_forward_gradient(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
    rank: int,
) -> tuple[NDArray[np.float64], float]:
    """Approximate the forward-only gradient using a rank-``r`` replay."""

    eigenvalues, eigenvectors, tail = _truncate_symmetric(
        sensitivity, rank
    )
    combined_vectors = replay_combinations(
        matvec, lanczos, eigenvectors
    )
    gradient = np.empty(len(derivative_matvecs), dtype=np.float64)
    for parameter, derivative_matvec in enumerate(derivative_matvecs):
        contractions = np.empty(rank, dtype=np.float64)
        for column in range(rank):
            vector = combined_vectors[:, column]
            applied = derivative_matvec(vector)
            contractions[column] = float(
                np.real(np.vdot(vector, applied))
            )
        gradient[parameter] = float(np.dot(eigenvalues, contractions))
    return gradient, tail


def _tangent_replay(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
    *,
    compute_tangent_norms: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Replay each parameter tangent once for all requested diagnostics."""

    correction = np.zeros(len(derivative_matvecs), dtype=np.float64)
    tangent_norms = (
        np.empty(len(derivative_matvecs), dtype=np.float64)
        if compute_tangent_norms
        else None
    )
    has_residual = (
        lanczos.residual_beta != 0.0
        and lanczos.residual_vector is not None
    )
    if not has_residual and not compute_tangent_norms:
        return correction, None
    last_row = np.asarray(sensitivity[-1, :], dtype=np.float64)
    residual_vector = lanczos.residual_vector

    for parameter, derivative_matvec in enumerate(derivative_matvecs):
        previous = np.zeros_like(lanczos.start_vector)
        current = lanczos.start_vector.copy()
        tangent_previous = np.zeros_like(lanczos.start_vector)
        tangent_current = np.zeros_like(lanczos.start_vector)
        beta_previous = 0.0
        tangent_beta_previous = 0.0
        eta = np.zeros(lanczos.steps, dtype=np.complex128)
        tangent_alphas = (
            np.empty(lanczos.steps, dtype=np.float64)
            if compute_tangent_norms
            else None
        )
        tangent_betas = (
            np.empty(max(0, lanczos.steps - 1), dtype=np.float64)
            if compute_tangent_norms
            else None
        )

        for iteration in range(lanczos.steps):
            if residual_vector is not None:
                eta[iteration] = np.vdot(
                    residual_vector, tangent_current
                )
            if iteration == lanczos.steps - 1 and not compute_tangent_norms:
                break

            applied_current = np.asarray(
                matvec(current), dtype=np.complex128
            )
            applied_tangent = np.asarray(
                matvec(tangent_current), dtype=np.complex128
            )
            derivative_current = np.asarray(
                derivative_matvec(current), dtype=np.complex128
            )
            alpha = lanczos.alphas[iteration]
            beta = lanczos.betas[iteration]

            residual = (
                applied_current
                - alpha * current
                - beta_previous * previous
            )
            next_vector = residual / beta
            tangent_alpha = float(
                np.real(
                    np.vdot(tangent_current, applied_current)
                    + np.vdot(current, derivative_current)
                    + np.vdot(current, applied_tangent)
                )
            )
            if tangent_alphas is not None:
                tangent_alphas[iteration] = tangent_alpha
            if iteration == lanczos.steps - 1:
                break
            tangent_residual = (
                derivative_current
                + applied_tangent
                - tangent_alpha * current
                - alpha * tangent_current
                - tangent_beta_previous * previous
                - beta_previous * tangent_previous
            )
            tangent_beta = float(
                np.real(np.vdot(next_vector, tangent_residual))
            )
            if tangent_betas is not None:
                tangent_betas[iteration] = tangent_beta
            tangent_next = (
                tangent_residual - tangent_beta * next_vector
            ) / beta

            previous, current = current, next_vector
            tangent_previous, tangent_current = (
                tangent_current,
                tangent_next,
            )
            beta_previous = beta
            tangent_beta_previous = tangent_beta

        if has_residual:
            correction[parameter] = float(
                2.0
                * lanczos.residual_beta
                * np.real(np.dot(last_row, eta))
            )
        if tangent_norms is not None:
            assert tangent_alphas is not None
            assert tangent_betas is not None
            tangent_matrix = np.diag(tangent_alphas)
            if tangent_betas.size:
                tangent_matrix += np.diag(tangent_betas, 1)
                tangent_matrix += np.diag(tangent_betas, -1)
            tangent_norms[parameter] = float(
                np.linalg.norm(tangent_matrix, ord=2)
            )
    return correction, tangent_norms


def residual_basis_correction(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> NDArray[np.float64]:
    """Compute the exact omitted basis term by parameterwise tangent replay.

    In exact arithmetic this returns
    ``2 beta_m Re(e_m.T G eta_a)`` for every real parameter ``a``.
    It is the difference between the derivative of the finite Lanczos
    approximation and the forward-only projected derivative.
    """

    correction, _ = _tangent_replay(
        matvec,
        lanczos,
        sensitivity,
        derivative_matvecs,
        compute_tangent_norms=False,
    )
    return correction


def tangent_tridiagonal_norms(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> NDArray[np.float64]:
    """Return ``||dT_m/dtheta_a||_2`` by parameterwise tangent replay.

    The calculation differentiates every retained diagonal coefficient and
    the first ``m-1`` off-diagonal coefficients.  It stores only the small
    derivative tridiagonal and a constant number of ambient vectors.
    """

    zero_sensitivity = np.zeros(
        (lanczos.steps, lanczos.steps), dtype=np.float64
    )
    _, norms = _tangent_replay(
        matvec,
        lanczos,
        zero_sensitivity,
        derivative_matvecs,
        compute_tangent_norms=True,
    )
    assert norms is not None
    return norms


def bessel_coefficient_tail_upper_bound(
    argument: float,
    first_degree: int,
    *,
    degree_squared: bool = False,
) -> float:
    """Bound a Bessel tail by positive, factorially decaying terms.

    For real ``z`` and integer ``k >= 1``, the power series gives

    ``|J_k(z)| <= (|z|/2)^k/k! * exp(z^2/(4(k+1)))``.

    This routine sums a geometric majorant of those terms.  With
    ``degree_squared=True`` it bounds ``sum k^2 |J_k(z)|``, the tail entering
    the operator-Markov bound for Fréchet derivatives of Chebyshev terms.
    """

    z = float(argument)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("argument must be finite and nonnegative")
    if first_degree < 1:
        raise ValueError("first_degree must be positive")
    if z == 0.0:
        return 0.0

    half = 0.5 * z
    accumulated = 0.0
    for degree in range(first_degree, 100000):
        log_majorant = (
            degree * math.log(half)
            - math.lgamma(degree + 1.0)
            + z * z / (4.0 * (degree + 1.0))
        )
        if log_majorant > math.log(np.finfo(float).max):
            raise OverflowError("Bessel-tail majorant exceeds float range")
        majorant = math.exp(log_majorant)
        weight = float(degree * degree) if degree_squared else 1.0
        if degree_squared:
            ratio_bound = half * (degree + 1.0) / (degree * degree)
        else:
            ratio_bound = half / (degree + 1.0)
        if ratio_bound < 1.0:
            remainder = weight * majorant / (1.0 - ratio_bound)
            return float(np.nextafter(accumulated + remainder, np.inf))
        accumulated += weight * majorant
    raise RuntimeError("failed to reach a decreasing Bessel majorant")


def _direct_tridiagonal_tangents(
    matvec: Matvec,
    lanczos: LanczosCoefficients,
    sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Contract ``dT_m`` with the projected sensitivity during replay.

    The routine differentiates the scalar three-term recurrence one parameter
    at a time.  It retains only the current and preceding primal and tangent
    vectors together with the small tridiagonal derivative.
    """

    gradients = np.empty(len(derivative_matvecs), dtype=np.float64)
    tangent_norms = np.empty(len(derivative_matvecs), dtype=np.float64)

    for parameter, derivative_matvec in enumerate(derivative_matvecs):
        previous = np.zeros_like(lanczos.start_vector)
        current = lanczos.start_vector.copy()
        tangent_previous = np.zeros_like(lanczos.start_vector)
        tangent_current = np.zeros_like(lanczos.start_vector)
        beta_previous = 0.0
        tangent_beta_previous = 0.0
        tangent_alphas = np.empty(lanczos.steps, dtype=np.float64)
        tangent_betas = np.empty(
            max(0, lanczos.steps - 1), dtype=np.float64
        )

        for iteration in range(lanczos.steps):
            applied_current = np.asarray(
                matvec(current), dtype=np.complex128
            )
            applied_tangent = np.asarray(
                matvec(tangent_current), dtype=np.complex128
            )
            derivative_current = np.asarray(
                derivative_matvec(current), dtype=np.complex128
            )
            alpha = lanczos.alphas[iteration]
            beta = lanczos.betas[iteration]
            tangent_alpha = float(
                np.real(
                    np.vdot(tangent_current, applied_current)
                    + np.vdot(current, derivative_current)
                    + np.vdot(current, applied_tangent)
                )
            )
            tangent_alphas[iteration] = tangent_alpha
            if iteration == lanczos.steps - 1:
                break

            residual = (
                applied_current
                - alpha * current
                - beta_previous * previous
            )
            if beta == 0.0:
                raise RuntimeError("cannot replay after Lanczos breakdown")
            next_vector = residual / beta
            tangent_residual = (
                derivative_current
                + applied_tangent
                - tangent_alpha * current
                - alpha * tangent_current
                - tangent_beta_previous * previous
                - beta_previous * tangent_previous
            )
            tangent_beta = float(
                np.real(np.vdot(next_vector, tangent_residual))
            )
            tangent_betas[iteration] = tangent_beta
            tangent_next = (
                tangent_residual - tangent_beta * next_vector
            ) / beta

            previous, current = current, next_vector
            tangent_previous, tangent_current = (
                tangent_current,
                tangent_next,
            )
            beta_previous = beta
            tangent_beta_previous = tangent_beta

        tangent_matrix = np.diag(tangent_alphas)
        if tangent_betas.size:
            tangent_matrix += np.diag(tangent_betas, 1)
            tangent_matrix += np.diag(tangent_betas, -1)
        gradients[parameter] = float(
            np.real(np.sum(sensitivity * tangent_matrix))
        )
        tangent_norms[parameter] = float(
            np.linalg.norm(tangent_matrix, ord=2)
        )

    return gradients, tangent_norms


@_finite_public_result
def direct_tridiagonal_replay_gradient(
    matvec: Matvec,
    start: ComplexVector,
    derivative_matvecs: Sequence[DerivativeMatvec],
    time: float,
    steps: int,
    *,
    derivative_operator_norms: NDArray[np.float64] | None = None,
    hamiltonian_norm_upper_bound: float | None = None,
) -> DirectReplayResult:
    """Differentiate the finite Lanczos objective without storing its basis.

    If ``T_m(theta)`` is the tridiagonal generated by the scalar Lanczos
    recurrence and ``G_m`` is the sensitivity of the projected survival
    probability to ``T_m``, the returned component is

    ``gradient[a] = trace(G_m.T @ dT_m/dtheta[a])``.

    Thus the method has no sensitivity-rank truncation error.  It requires one
    ordinary Lanczos pass and one parameterwise tangent replay, uses a number
    of ambient vectors independent of ``steps``, and accesses the Hamiltonian
    and every parameter derivative only through matrix-vector products.
    Supplying a verified upper bound on ``||A||_2`` adds the same
    Chebyshev--Bessel truncation bound used for the unprojected survival
    gradient; it does not cover floating-point roundoff or loss of Lanczos
    orthogonality.
    """

    coefficients = low_memory_lanczos(matvec, start, steps)
    probability, sensitivity = survival_probability_sensitivity(
        coefficients.start_norm,
        coefficients.tridiagonal,
        time,
    )
    gradient, tangent_norms = _direct_tridiagonal_tangents(
        matvec,
        coefficients,
        sensitivity,
        derivative_matvecs,
    )

    if derivative_operator_norms is None:
        operator_norms = np.ones(
            len(derivative_matvecs), dtype=np.float64
        )
    else:
        operator_norms = np.asarray(
            derivative_operator_norms, dtype=np.float64
        )
        if operator_norms.shape != (len(derivative_matvecs),):
            raise ValueError("derivative_operator_norms has the wrong shape")
        if np.any(operator_norms < 0.0):
            raise ValueError("derivative operator norms must be nonnegative")

    component_bound = None
    vector_bound = None
    value_tail = None
    derivative_tail = None
    if hamiltonian_norm_upper_bound is not None:
        matrix_norm_bound = float(hamiltonian_norm_upper_bound)
        if not np.isfinite(matrix_norm_bound) or matrix_norm_bound <= 0.0:
            raise ValueError(
                "hamiltonian_norm_upper_bound must be finite and positive"
            )
        projected_norm = float(
            np.linalg.norm(coefficients.tridiagonal, ord=2)
        )
        norm_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, projected_norm, matrix_norm_bound
        )
        if projected_norm > matrix_norm_bound + norm_tolerance:
            raise ValueError(
                "hamiltonian_norm_upper_bound is smaller than a Ritz value"
            )
        scaled_norm = abs(float(time)) * matrix_norm_bound
        first_omitted_degree = 2 * coefficients.steps
        value_tail = 2.0 * bessel_coefficient_tail_upper_bound(
            scaled_norm, first_omitted_degree
        )
        derivative_tail = (
            2.0
            / matrix_norm_bound
            * bessel_coefficient_tail_upper_bound(
                scaled_norm,
                first_omitted_degree,
                degree_squared=True,
            )
        )
        norm_fourth = coefficients.start_norm**4
        component_bound = (
            4.0
            * norm_fourth
            * abs(float(time))
            * operator_norms
            * value_tail
            + 2.0
            * norm_fourth
            * (operator_norms + tangent_norms)
            * derivative_tail
        )
        vector_bound = float(np.linalg.norm(component_bound))

    dimension = coefficients.start_vector.size
    projected_vector_equivalents = (
        4.0 * coefficients.steps**2 / max(dimension, 1)
    )
    return DirectReplayResult(
        probability=float(probability),
        gradient=np.asarray(gradient, dtype=np.float64),
        steps=coefficients.steps,
        estimated_peak_vector_equivalents=float(
            10.0 + projected_vector_equivalents
        ),
        hamiltonian_matvecs=int(
            coefficients.steps
            + 2 * len(derivative_matvecs) * coefficients.steps
        ),
        derivative_matvecs=int(
            len(derivative_matvecs) * coefficients.steps
        ),
        tangent_tridiagonal_norms=tangent_norms,
        krylov_component_error_bound=component_bound,
        krylov_vector_error_bound=vector_bound,
        chebyshev_value_tail=value_tail,
        chebyshev_lipschitz_tail=derivative_tail,
    )


@_finite_public_result
def error_bounded_survival_gradient(
    matvec: Matvec,
    start: ComplexVector,
    derivative_matvecs: Sequence[DerivativeMatvec],
    time: float,
    steps: int,
    rank: int,
    *,
    derivative_operator_norms: NDArray[np.float64] | None = None,
    hamiltonian_norm_upper_bound: float | None = None,
    include_residual_correction: bool = True,
) -> GradientResult:
    """Compute a low-memory gradient and optional total exact-arithmetic bound.

    Without ``hamiltonian_norm_upper_bound`` the returned certificate has the
    original scope: compression relative to the finite Lanczos objective.
    Supplying a verified upper bound on ``||A||_2`` additionally bounds the
    Krylov truncation error of the survival-probability gradient by
    differentiated degree-``2m-1`` moment matching and Chebyshev--Bessel
    tails.  Roundoff and loss of Lanczos orthogonality remain outside scope.
    """

    lanczos = low_memory_lanczos(matvec, start, steps)
    actual_rank = min(rank, lanczos.steps)
    if actual_rank < 1:
        raise ValueError("rank must be positive")
    probability, sensitivity = survival_probability_sensitivity(
        lanczos.start_norm,
        lanczos.tridiagonal,
        time,
    )
    compressed, tail = compressed_forward_gradient(
        matvec,
        lanczos,
        sensitivity,
        derivative_matvecs,
        actual_rank,
    )
    tangent_norms = None
    if hamiltonian_norm_upper_bound is None:
        correction = residual_basis_correction(
            matvec,
            lanczos,
            sensitivity,
            derivative_matvecs,
        )
    else:
        correction, tangent_norms = _tangent_replay(
            matvec,
            lanczos,
            sensitivity,
            derivative_matvecs,
            compute_tangent_norms=True,
        )
    gradient = (
        compressed + correction
        if include_residual_correction
        else compressed
    )

    if derivative_operator_norms is None:
        operator_norms = np.ones(
            len(derivative_matvecs), dtype=np.float64
        )
    else:
        operator_norms = np.asarray(
            derivative_operator_norms, dtype=np.float64
        )
        if operator_norms.shape != (len(derivative_matvecs),):
            raise ValueError(
                "derivative_operator_norms has the wrong shape"
            )
        if np.any(operator_norms < 0):
            raise ValueError(
                "derivative operator norms must be nonnegative"
            )

    if include_residual_correction:
        component_bound = tail * operator_norms
        vector_bound = float(tail * np.linalg.norm(operator_norms))
    else:
        component_bound = np.abs(correction) + tail * operator_norms
        vector_bound = float(
            np.linalg.norm(correction)
            + tail * np.linalg.norm(operator_norms)
        )

    krylov_component_bound = None
    total_component_bound = None
    total_vector_bound = None
    value_tail = None
    derivative_tail = None
    if hamiltonian_norm_upper_bound is not None:
        matrix_norm_bound = float(hamiltonian_norm_upper_bound)
        if not np.isfinite(matrix_norm_bound) or matrix_norm_bound < 0.0:
            raise ValueError(
                "hamiltonian_norm_upper_bound must be finite and nonnegative"
            )
        projected_norm = float(
            np.linalg.norm(lanczos.tridiagonal, ord=2)
        )
        norm_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, projected_norm, matrix_norm_bound
        )
        if projected_norm > matrix_norm_bound + norm_tolerance:
            raise ValueError(
                "hamiltonian_norm_upper_bound is smaller than a Ritz value"
            )
        assert tangent_norms is not None
        if matrix_norm_bound == 0.0:
            raise ValueError(
                "hamiltonian_norm_upper_bound must be positive for the "
                "Chebyshev derivative certificate"
            )
        scaled_norm = abs(float(time)) * matrix_norm_bound
        # Scalar Gauss--Lanczos moment matching is exact through degree 2m-1.
        # The omitted Chebyshev--Bessel expansion starts at degree 2m.  The
        # operator-Markov inequality contributes k^2 to its derivative tail.
        first_omitted_degree = 2 * lanczos.steps
        value_tail = 2.0 * bessel_coefficient_tail_upper_bound(
            scaled_norm, first_omitted_degree
        )
        derivative_tail = (
            2.0
            / matrix_norm_bound
            * bessel_coefficient_tail_upper_bound(
                scaled_norm,
                first_omitted_degree,
                degree_squared=True,
            )
        )
        norm_fourth = lanczos.start_norm**4
        absolute_time = abs(float(time))
        krylov_component_bound = (
            4.0
            * norm_fourth
            * absolute_time
            * operator_norms
            * value_tail
            + 2.0
            * norm_fourth
            * (operator_norms + tangent_norms)
            * derivative_tail
        )
        total_component_bound = component_bound + krylov_component_bound
        total_vector_bound = float(np.linalg.norm(total_component_bound))

    # Complex-vector storage during parameterwise tangent replay is bounded
    # independently of m. Small O(m^2) projected matrices are converted to
    # vector equivalents for transparent comparison with an n-by-m basis.
    dimension = start.size
    projected_vector_equivalents = (
        6.0 * lanczos.steps**2 / max(dimension, 1)
    )
    peak_vectors = float(
        max(actual_rank + 5, 10) + projected_vector_equivalents
    )

    return GradientResult(
        probability=probability,
        gradient=gradient,
        compressed_forward_gradient=compressed,
        residual_correction=correction,
        component_error_bound=component_bound,
        vector_error_bound=vector_bound,
        compression_tail_nuclear=tail,
        rank=actual_rank,
        steps=lanczos.steps,
        estimated_peak_vector_equivalents=peak_vectors,
        krylov_component_error_bound=krylov_component_bound,
        total_component_error_bound=total_component_bound,
        total_vector_error_bound=total_vector_bound,
        tangent_tridiagonal_norms=tangent_norms,
        chebyshev_value_tail=value_tail,
        chebyshev_lipschitz_tail=derivative_tail,
    )


@dataclass(frozen=True)
class LanczosTrace:
    """Minimal coefficients used by the shared-subspace replay algorithm."""

    alphas: NDArray[np.float64]
    betas: NDArray[np.float64]
    residual_beta: float
    start_norm: float

    @property
    def steps(self) -> int:
        return self.alphas.size

    @property
    def tridiagonal(self) -> NDArray[np.float64]:
        matrix = np.diag(self.alphas)
        if self.betas.size:
            matrix += np.diag(self.betas, 1) + np.diag(self.betas, -1)
        return matrix


@dataclass(frozen=True)
class LowMemoryDiagnostics:
    """Compression and large-vector storage diagnostics."""

    shared_rank: int
    stored_vector_equivalents: int
    full_basis_vector_equivalents: int
    cosine_tail_frobenius: float
    sine_tail_frobenius: float
    residual_beta: float


def lanczos_trace(
    matvec: Matvec,
    start: ComplexVector,
    steps: int,
    *,
    breakdown_tolerance: float = 1e-14,
) -> LanczosTrace:
    """Run unreorthogonalized Lanczos while storing three large vectors."""

    coefficients = low_memory_lanczos(
        matvec,
        start,
        steps,
        breakdown_tolerance=breakdown_tolerance,
    )
    return LanczosTrace(
        alphas=coefficients.alphas,
        betas=coefficients.betas[: max(0, coefficients.steps - 1)],
        residual_beta=coefficients.residual_beta,
        start_norm=coefficients.start_norm,
    )


def replay_projected_basis(
    matvec: Matvec,
    start: ComplexVector,
    trace: LanczosTrace,
    modes: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Accumulate ``V @ modes`` by replaying recorded coefficients."""

    mode_matrix = np.asarray(modes, dtype=np.float64)
    if mode_matrix.ndim != 2 or mode_matrix.shape[0] != trace.steps:
        raise ValueError("modes must have one row per Lanczos step")

    start_vector = np.asarray(start, dtype=np.complex128)
    current = start_vector / trace.start_norm
    previous = np.zeros_like(current)
    projected = np.outer(current, mode_matrix[0, :])

    for iteration in range(trace.steps - 1):
        work = np.asarray(matvec(current), dtype=np.complex128)
        work -= trace.alphas[iteration] * current
        if iteration:
            work -= trace.betas[iteration - 1] * previous
        following = work / trace.betas[iteration]
        projected += np.outer(
            following, mode_matrix[iteration + 1, :]
        )
        previous, current = current, following
    return projected


def _projected_value_and_sensitivity(
    trace: LanczosTrace,
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    derivative: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[float, NDArray[np.float64]]:
    eigenvalues, eigenvectors = np.linalg.eigh(trace.tridiagonal)
    c = eigenvectors[0, :]
    value = float(
        trace.start_norm**2 * np.dot(c**2, function(eigenvalues))
    )
    kernel = divided_difference(eigenvalues, function, derivative)
    sensitivity = (
        trace.start_norm**2
        * eigenvectors
        @ (np.outer(c, c) * kernel)
        @ eigenvectors.T
    )
    sensitivity = np.asarray(
        0.5 * (sensitivity + sensitivity.T), dtype=np.float64
    )
    return value, sensitivity


def _dominant_modes(
    sensitivity: NDArray[np.float64], rank: int
) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(sensitivity)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvectors[:, order[: min(rank, sensitivity.shape[0])]]


def _contract_shared_subspace(
    compressed_basis: NDArray[np.complex128],
    small_sensitivity: NDArray[np.float64],
    derivative_matvecs: Sequence[DerivativeMatvec],
) -> NDArray[np.float64]:
    gradients = np.empty(len(derivative_matvecs), dtype=np.float64)
    for index, derivative_matvec in enumerate(derivative_matvecs):
        applied = np.column_stack(
            [
                derivative_matvec(compressed_basis[:, column])
                for column in range(compressed_basis.shape[1])
            ]
        )
        projected_derivative = compressed_basis.conj().T @ applied
        gradients[index] = float(
            np.real(
                np.sum(small_sensitivity * projected_derivative.T)
            )
        )
    return gradients


@_finite_public_result
def low_memory_survival_probability_gradient(
    matvec: Matvec,
    start: ComplexVector,
    derivative_matvecs: Sequence[DerivativeMatvec],
    time: float,
    steps: int,
    rank: int,
) -> tuple[
    float,
    NDArray[np.float64],
    LowMemoryDiagnostics,
]:
    """Two-pass shared-subspace forward-only survival gradient."""

    if rank < 1:
        raise ValueError("rank must be positive")
    trace = lanczos_trace(matvec, start, steps)
    cosine = lambda x: np.cos(time * x)
    cosine_derivative = lambda x: -time * np.sin(time * x)
    sine = lambda x: np.sin(time * x)
    sine_derivative = lambda x: time * np.cos(time * x)

    real_amplitude, cosine_sensitivity = (
        _projected_value_and_sensitivity(
            trace, cosine, cosine_derivative
        )
    )
    minus_imaginary_amplitude, sine_sensitivity = (
        _projected_value_and_sensitivity(trace, sine, sine_derivative)
    )
    candidate_modes = np.column_stack(
        (
            _dominant_modes(cosine_sensitivity, rank),
            _dominant_modes(sine_sensitivity, rank),
        )
    )
    shared_modes, _ = np.linalg.qr(candidate_modes, mode="reduced")
    shared_rank = min(shared_modes.shape[1], trace.steps)
    shared_modes = shared_modes[:, :shared_rank]
    projector = shared_modes @ shared_modes.T
    compressed_cosine = (
        shared_modes.T @ cosine_sensitivity @ shared_modes
    )
    compressed_sine = (
        shared_modes.T @ sine_sensitivity @ shared_modes
    )
    cosine_tail = float(
        np.linalg.norm(
            cosine_sensitivity
            - projector @ cosine_sensitivity @ projector,
            ord="fro",
        )
    )
    sine_tail = float(
        np.linalg.norm(
            sine_sensitivity
            - projector @ sine_sensitivity @ projector,
            ord="fro",
        )
    )

    compressed_basis = replay_projected_basis(
        matvec, start, trace, shared_modes
    )
    cosine_gradient = _contract_shared_subspace(
        compressed_basis, compressed_cosine, derivative_matvecs
    )
    sine_gradient = _contract_shared_subspace(
        compressed_basis, compressed_sine, derivative_matvecs
    )
    probability = real_amplitude**2 + minus_imaginary_amplitude**2
    gradient = (
        2.0 * real_amplitude * cosine_gradient
        + 2.0 * minus_imaginary_amplitude * sine_gradient
    )
    diagnostics = LowMemoryDiagnostics(
        shared_rank=shared_rank,
        stored_vector_equivalents=shared_rank + 3,
        full_basis_vector_equivalents=trace.steps,
        cosine_tail_frobenius=cosine_tail,
        sine_tail_frobenius=sine_tail,
        residual_beta=trace.residual_beta,
    )
    return float(probability), gradient, diagnostics
