"""Matrix-free Krylov tools for coherent quantum-error calibration."""

from .lanczos import LanczosResult, hermitian_lanczos
from .block_replay import BlockReplayResult, parameter_blocked_replay_gradient
from .low_memory import (
    DirectReplayResult,
    GradientResult,
    LanczosCoefficients,
    bessel_coefficient_tail_upper_bound,
    direct_tridiagonal_replay_gradient,
    error_bounded_survival_gradient,
    low_memory_lanczos,
    tangent_tridiagonal_norms,
)
from .low_memory import (
    LanczosTrace,
    LowMemoryDiagnostics,
    lanczos_trace,
    low_memory_survival_probability_gradient,
)
from .pauli import PauliHamiltonian, PauliString
from .projected import (
    projected_quadratic_form,
    projected_sensitivity,
    survival_probability_gradient,
)

__all__ = [
    "LanczosResult",
    "LanczosCoefficients",
    "GradientResult",
    "DirectReplayResult",
    "BlockReplayResult",
    "LanczosTrace",
    "LowMemoryDiagnostics",
    "PauliHamiltonian",
    "PauliString",
    "hermitian_lanczos",
    "low_memory_lanczos",
    "error_bounded_survival_gradient",
    "direct_tridiagonal_replay_gradient",
    "parameter_blocked_replay_gradient",
    "bessel_coefficient_tail_upper_bound",
    "tangent_tridiagonal_norms",
    "lanczos_trace",
    "low_memory_survival_probability_gradient",
    "projected_quadratic_form",
    "projected_sensitivity",
    "survival_probability_gradient",
]

__version__ = "2.0.0"
