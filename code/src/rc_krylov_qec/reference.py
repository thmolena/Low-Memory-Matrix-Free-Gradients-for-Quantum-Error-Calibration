"""Dense validation oracles for small systems only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, expm_frechet


def dense_survival_probability_gradient(
    hamiltonian: NDArray[np.complex128],
    derivative_matrices: Sequence[NDArray[np.complex128]],
    state: NDArray[np.complex128],
    time: float,
) -> tuple[float, NDArray[np.float64]]:
    """Exact small-system reference based on dense Fréchet derivatives."""

    generator = -1j * time * hamiltonian
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        evolution = expm(generator)
        amplitude = np.vdot(state, evolution @ state)
        probability = float(abs(amplitude) ** 2)

        gradient = np.empty(len(derivative_matrices), dtype=np.float64)
        for index, derivative_matrix in enumerate(derivative_matrices):
            derivative_evolution = expm_frechet(
                generator,
                -1j * time * derivative_matrix,
                compute_expm=False,
            )
            derivative_amplitude = np.vdot(
                state, derivative_evolution @ state
            )
            gradient[index] = 2.0 * np.real(
                np.conjugate(amplitude) * derivative_amplitude
            )
    if not np.isfinite(probability) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("dense survival reference is nonfinite")
    return probability, gradient
